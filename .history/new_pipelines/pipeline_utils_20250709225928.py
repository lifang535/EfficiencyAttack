import os
import sys
import pdb
import json
import torch
import random
import argparse
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
import torch.multiprocessing as mp
from torch.utils.data import Subset

sys.path.append("../")
import utils
import model_zoo
utils.set_all_seeds(0)


def ms_coco_init(num_samples=None):
    coco_data = load_dataset("detection-datasets/coco", split="val")
    if num_samples is None:
        return coco_data
    random_indices = random.sample(range(len(coco_data)), num_samples)  
    coco_data = coco_data.select(random_indices)
    return coco_data
    
    
def object_detection_init(ckpt=0, num_q=200, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = model_zoo.load_from_pretrained(
        ckpt=ckpt,
        num_q=num_q,
        device=device,
    )
    return model, processor


def object_detection_preprocess(example, processor, device=None):
    image_id, img, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(example)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_tensor = processor(img, return_tensors="pt")["pixel_values"].to(device)
    img_tensor = utils.denormalize(img_tensor)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0).to(device)
    return image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area


def get_target_size(img_tensor):
    b, _, h, w = img_tensor.shape
    minimum_size = 128

    length = max(h, w)
    div = length // minimum_size
    padded_length = min((div + 1) * minimum_size, 640)
    target_size =  [torch.Size([padded_length, padded_length])]
    if b == 1:
        return target_size
    else:
        return [torch.Size([padded_length, padded_length]) for _ in range(b)]


def object_detection_inference(model, processor, img_tensor, target_size, thres, device=None):
    raw = model(img_tensor)
    results = processor.post_process_object_detection(
        raw, 
        target_sizes=target_size, 
        threshold=thres, 
    )
    if len(results) == 0:
        return None
    else:
        return results
    
    
def get_num_dets(result):
    num_dets = 0
    for res in result:
        res_boxes = res.get("boxes", None)
        if res_boxes is not None:
            num_dets += len(res_boxes)
    return num_dets
        

def cropper(img_tensor, prediction_data, device):
    assert img_tensor.ndim == 4, "Image tensor should have 4 dimensions (batch_size, channels, height, width)"
    
    boxes = prediction_data.get("boxes", None)
    crops = []
    if boxes is None or img_tensor is None:
        return crops
    
    for box in boxes:
        if box is None:
            continue
        
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_tensor.shape[-1], x2)
        y2 = min(img_tensor.shape[-2], y2)
        
        if x2 > x1 and y2 > y1:  # Valid crop
            crop = img_tensor[:, :, y1:y2, x1:x2].to(device)
            crops.append(crop)
    return crops


def crop_interpolate_pad(img_tensor, res, device, interpolate_size=(128, 128)):
    boxes = res.get("boxes", None)
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_tensor.shape[-1], x2)
        y2 = min(img_tensor.shape[-2], y2)

        if x2 > x1 and y2 > y1:
            crop = img_tensor[:, :, y1:y2, x1:x2].to(device)  # [B, C, H, W]
            _, _, h, w = crop.shape

            scale = min(interpolate_size[0] / h, interpolate_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)

            resized_crop = F.interpolate(
                crop,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

            pad_h = interpolate_size[0] - new_h
            pad_w = interpolate_size[1] - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            padded = F.pad(
                resized_crop,
                (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
                mode='constant',
                value=0
            )
            crops.append(padded)  # [B, C, 128, 128]
    return crops


def get_cls_loss_target(prob, target_idx=None):
    target_tensor = torch.zeros_like(prob).to(prob)
    if isinstance(target_idx, int):
        target_tensor[:, target_idx] = 1.0
    elif isinstance(target_idx, list):
        for i in target_idx:
            target_tensor[:, i] = 1.0
    elif target_idx is None:
        target_tensor = torch.ones_like(prob)
    return target_tensor


def calculate_cls_loss(raw, target_idx=None):
    probs = F.sigmoid(raw.logits)
    target_tensor = torch.zeros_like(probs).to(probs)
    
    with torch.no_grad():
        if isinstance(target_idx, int):
            target_tensor[:, :, target_idx] = 1.0
        elif isinstance(target_idx, list):
            for i in target_idx:
                target_tensor[:, :, i] = 1.0
        elif target_idx is None:
            target_tensor = torch.ones_like(probs)
            
    loss = 1.0 * F.mse_loss(probs, target_tensor, reduction='sum') / max(probs.shape[1], 1)
    return loss



def calculate_iou_loss(res, patch_bbox):
    """
    Calculates a vectorized, numerically stable IoU loss.
    """
    # Detected boxes from the model output
    det_boxes = res.get("boxes")

    # If no boxes are detected, return a loss of 0
    if det_boxes is None or len(det_boxes) == 0:
        return torch.tensor(0.0, device=patch_bbox.device)

    # Unpack patch coordinates and ensure it's in the correct format [x1, y1, x2, y2]
    px1, py1, px2, py2 = patch_bbox
    patch_area = (px2 - px1) * (py2 - py1)

    # Unpack detected box coordinates
    x1, y1, x2, y2 = det_boxes.T
    box_area = (x2 - x1) * (y2 - y1)

    # --- Vectorized Intersection Calculation ---
    # Find the top-left corner of the intersection area
    inter_x1 = torch.max(x1, px1)
    inter_y1 = torch.max(y1, py1)
    # Find the bottom-right corner of the intersection area
    inter_x2 = torch.min(x2, px2)
    inter_y2 = torch.min(y2, py2)

    # Calculate intersection area, clamping at 0 for non-overlapping boxes
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # --- Vectorized Union Calculation ---
    union_area = box_area + patch_area - inter_area

    # Calculate IoU and the corresponding loss (1 - IoU)
    iou = inter_area / (union_area + 1e-6) # Add epsilon for numerical stability
    iou_loss = 1.0 - iou
    
    # Return the mean loss over all detected boxes
    return iou_loss.mean()


def calculate_distance_loss(res, patch_bbox):
    px1, py1, px2, py2 = patch_bbox
    distance_loss = torch.tensor(0.0).to(patch_bbox)
    cnt = 0
    for box in res.get("boxes", []):
        x1, y1, x2, y2 = box.int()

        if x2 > x1 and y2 > y1:
            left_margin = max(0, x1 - px1)
            right_margin = max(0, x2 - px2)
            top_margin = max(0, y1 - py1)
            bottom_margin = max(0, y2 - py2)
            total_distance = left_margin + right_margin + top_margin + bottom_margin
            distance_loss += total_distance / (px1 + py1)
            cnt += 1
            
    return distance_loss / max(cnt, 1)  # Avoid division by zero
    