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


def crop_interpolate_pad(img_tensor, res, device):
    boxes = res.get("boxes", None)
    crops = []
    interpolate_size = get_target_size(img_tensor)[0]
    for box in boxes:
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_tensor.shape[-1], x2)
        y2 = min(img_tensor.shape[-2], y2)
        if x2 > x1 and y2 > y1:
            crop = img_tensor[:, :, y1:y2, x1:x2].to(device)
            interpolate_size = get_target_size(crop)[0]
            interpolated_img_tensor = F.interpolate(
                crop, 
                size=(interpolate_size[0], interpolate_size[1]), 
                mode='bilinear', 
                align_corners=False
            )
            padded_img_tensor = F.pad(
                interpolated_img_tensor,
                (0, 640 - interpolate_size[1],  # pad width: (left=0, right=pad_w)
                 0, 640 - interpolate_size[0]),  # pad height: (top=0, bottom=pad_h)
                mode='constant',
                value=0.0
            )
            crops.append(padded_img_tensor)
    return crops


def get_cls_loss_target(prob, target_idx=None):
    target_tensor = torch.zeros_like(prob)
    
    if isinstance(target_idx, int):
        target_tensor[:, target_idx] = 1.0
    elif isinstance(target_idx, list):
        for i in target_idx:
            target_tensor[:, i] = 1.0
    elif target_idx is None:
        target_tensor = torch.ones_like(prob)
    return target_tensor


def object_detection_loss(raw, target_idx=None):
    probs = F.sigmoid(raw.logits)
    target_tensor = torch.zeros_like(probs).to(probs)
    
    with torch.no_grad():
        if isinstance(target_idx, int):
            target_tensor[:, target_idx] = 1.0
        elif isinstance(target_idx, list):
            for i in target_idx:
                target_tensor[:, i] = 1.0
        elif target_idx is None:
            target_tensor = torch.ones_like(probs)
            
    loss = 1.0 * F.mse_loss(probs, target_tensor, reduction='sum') / ((probs.shape[0] + 1) * (probs.shape[1] + 1))
    return loss

