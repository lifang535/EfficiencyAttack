import os
import sys
import pdb
import json
import math
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

from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_ZOO = {
    0: "PekingU/rtdetr_r50vd",
    1: "PekingU/rtdetr_r50vd_coco_o365",
    2: "PekingU/rtdetr_v2_r50vd"
}

def load_from_pretrained(ckpt=None, num_q=1000, device=None):
    print(f"loading model on device: {device}")
    """Load a model and image processor from the model zoo."""
    ckpt, ckpt_id = get_model_name(ckpt)

    print(f"Initializing checkpoint: {ckpt}, ID: {ckpt_id}")

    # Load the correct model based on the checkpoint ID
    if ckpt_id in {0, 1}:
        model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device)
    elif ckpt_id == 2:
        model = RTDetrV2ForObjectDetection.from_pretrained(ckpt).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint ID: {ckpt_id}")

    image_processor = RTDetrImageProcessor.from_pretrained(ckpt)
    model = model.eval()
    model.config.num_queries = num_q

    return model, image_processor

def get_model_name(ckpt):
    """Retrieve model name from the zoo given an integer ID or directly validate a string checkpoint."""
    if isinstance(ckpt, int):
        if ckpt not in MODEL_ZOO:
            raise ValueError(f"Invalid checkpoint ID: {ckpt}. Available IDs: {list(MODEL_ZOO.keys())}")
        return MODEL_ZOO[ckpt], ckpt
    elif isinstance(ckpt, str) and ckpt in MODEL_ZOO.values():
        return ckpt, list(MODEL_ZOO.keys())[list(MODEL_ZOO.values()).index(ckpt)]
    else:
        raise ValueError(
            f"Invalid checkpoint: {ckpt}. \n"
            f"Available options:\n"
            f"IDs: {list(MODEL_ZOO.keys())}\n"
            f"Names: {list(MODEL_ZOO.values())}\n"
            "Please choose a valid ID or checkpoint name."
        )

def ms_coco_init(num_samples=None):
    coco_data = load_dataset("detection-datasets/coco", split="val")
    if num_samples is None:
        return coco_data
    random_indices = random.sample(range(len(coco_data)), num_samples)  
    coco_data = coco_data.select(random_indices)
    return coco_data


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


def crop_and_pad(img_tensor, res, device, pad_size=(128, 128)):
    boxes = res.get("boxes", None)
    crops = []

    max_w, max_h = pad_size

    for box in boxes:
        x1, y1, x2, y2 = box.detach().int()
        w, h = x2 - x1, y2 - y1
        
        if w > max_w or h > max_h:
            continue
        
        crop = img_tensor[:, :, y1:y2, x1:x2].to(device) 
        pad_left = ./
        

        
    
def crop_interpolate_pad(img_tensor, res, device, interpolate_size=(128, 128)):
    boxes = res.get("boxes", None)
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box.detach().int()
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
    if raw.logits.requires_grad == False:
        return torch.tensor(0.0, device=raw.logits.device, requires_grad=True)
    
    probs = F.sigmoid(raw.logits)
    target_tensor = torch.zeros_like(probs).to(probs)
    
    with torch.no_grad():
        if isinstance(target_idx, int):
            scale = 1.0
            target_tensor[:, :, target_idx] = 1.0
        elif isinstance(target_idx, list):
            scale = 1.0 / len(target_idx)
            for i in target_idx:
                target_tensor[:, :, i] = 1.0
        elif target_idx is None:
            scale = 1.0 / probs.shape[2]
            target_tensor = torch.ones_like(probs)
            
    loss = scale * F.mse_loss(probs, target_tensor, reduction='sum') / max(probs.shape[1], 1)
    return loss



def calculate_iou_loss(bbox_pred, patch_bbox):
    # --- 1. Input Setup and Box Extraction ---
    
    # Detected boxes from the model output, assuming format [N, 4] where N is num boxes
    det_boxes = bbox_pred[0]

    # If no boxes are detected, there is no loss to compute.
    if det_boxes is None or len(det_boxes) == 0:
        return torch.tensor(0.0, device=patch_bbox.device, requires_grad=True)

    # Unpack patch coordinates.
    # IMPORTANT: The user's original code included a division by 640.
    # This implies the model's output boxes are normalized (e.g., to [0, 1]),
    # while the patch_bbox is in pixel coordinates. We will keep this normalization
    # step, assuming it's required to match the coordinate systems.
    px1, py1, px2, py2 = patch_bbox 
    
    # Get basic properties of the patch (ground truth)
    p_w = px2 - px1
    p_h = py2 - py1
    patch_area = p_w * p_h
    
    # Unpack detected box coordinates
    x1, y1, x2, y2 = det_boxes.T
    
    # Get basic properties of the detected boxes
    b_w = x2 - x1
    b_h = y2 - y1
    box_area = b_w * b_h

    # --- 2. IoU Calculation (Same as before) ---
    
    # Calculate intersection area
    inter_x1 = torch.max(x1, px1)
    inter_y1 = torch.max(y1, py1)
    inter_x2 = torch.min(x2, px2)
    inter_y2 = torch.min(y2, py2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Calculate union area
    union_area = box_area + patch_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6) # Add epsilon for stability

    # --- 3. DIoU Penalty: Center Point Distance ---
    # This term penalizes the distance between the centers of the patch and detected boxes.
    # It provides a gradient even when IoU is zero.
    
    # Find the coordinates of the smallest enclosing box that contains both the patch and the detected boxes
    enclose_x1 = torch.min(x1, px1)
    enclose_y1 = torch.min(y1, py1)
    enclose_x2 = torch.max(x2, px2)
    enclose_y2 = torch.max(y2, py2)

    # Calculate the squared diagonal length of this enclosing box
    enclose_c2 = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + 1e-6

    # Calculate the center points of the patch and the detected boxes
    p_cx = (px1 + px2) / 2
    p_cy = (py1 + py2) / 2
    b_cx = (x1 + x2) / 2
    b_cy = (y1 + y2) / 2

    # Calculate the squared distance between the center points
    center_dist_sq = (b_cx - p_cx)**2 + (b_cy - p_cy)**2
    
    # The DIoU penalty term
    diou_penalty = center_dist_sq / enclose_c2

    # --- 4. CIoU Penalty: Aspect Ratio Consistency ---
    # This term penalizes inconsistencies in aspect ratio between the patch and detected boxes.
    
    # Calculate the aspect ratio term 'v'
    # Use atan for aspect ratio to keep the range constrained
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(p_w / (p_h + 1e-6)) - torch.atan(b_w / (b_h + 1e-6)), 2)

    # Calculate the trade-off parameter 'alpha'.
    # This is done with no_grad() as alpha is a non-gradient term.
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + 1e-6)

    # The final aspect ratio penalty
    aspect_ratio_penalty = alpha * v

    # --- 5. Final CIoU Loss Calculation ---
    
    # The complete loss is a combination of all three components
    ciou_loss = 1.0 - iou + diou_penalty + aspect_ratio_penalty
    
    # Return the mean loss over all detected boxes
    return ciou_loss.mean()


def calculate_distance_loss(bbox_pred, patch_bbox):
    if bbox_pred.ndim == 3:
        bbox_pred = bbox_pred[0]
    
    if bbox_pred is None or len(bbox_pred) == 0:
        return torch.tensor(0.0, device=patch_bbox.device, requires_grad=True)
    
    px1, py1, px2, py2 = patch_bbox
    x1, y1, x2, y2 = bbox_pred.T

    left_diff = torch.abs(x1 - px1)
    right_diff = torch.abs(x2 - px2)
    top_diff = torch.abs(y1 - py1)
    bottom_diff = torch.abs(y2 - py2)
    
    return ((left_diff + right_diff + top_diff + bottom_diff) / (px1 + px2 + py1 + py2)).mean()
    