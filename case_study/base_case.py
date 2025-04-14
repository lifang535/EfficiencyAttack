import sys
sys.path.append("..")
from datasets import load_dataset
import dataset
from utils import set_all_seeds
import torch
import random 
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import utils 
import os 
import pdb
from model_zoo import load_from_pretrained
from datasets import concatenate_datasets
from datetime import datetime
set_all_seeds(0)
from torchvision import transforms
from transformers import AutoModelForCausalLM # microsoft/git-base
from transformers import AutoProcessor
import torch.nn.functional as F
import numpy as np

def load_ms_coco_dataset(val_size):
    coco_data = load_dataset("detection-datasets/coco", split="val")
    if val_size:
        """val_size: number of samples"""
        random_indices = random.sample(range(len(coco_data)), val_size)
        return coco_data.select(random_indices)
    else:
        return coco_data
        

def load_rt_detr(model_id, num_q=1000, device=None):
    """
    Load the RT-DETR model from Hugging Face
        model_id: 0, 1, 2
        num_q: number of queries
        device: device to load the model on
        
    Usage of model and processor 
    
        for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__()):
            image_id, image, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(example)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_tensor = processor(image, return_tensors="pt")["pixel_values"].to(device)
            denorm_image_tensor = utils.denormalize(image_tensor)
            target_size = [image_tensor.shape[2:] for _ in range(1)]
            
            output = model(image_tensor)
            logits = output.logits[0]
            probs = F.sigmoid(logits)
            
            _output = processor.post_process_object_detection(output, 
                                                              threshold = 0.25, 
                                                              target_sizes = target_size)[0]
                                                              
            scores, labels, boxes = _output["scores"],  _output["labels"], _output["boxes"]

    """
    model, processor = load_from_pretrained(model_id, num_q=num_q, device=device)
    return model, processor

def object_detection_loss_target(probs, target_idx):
    """
    target_idx:
        int: index of the target class
        list: list of indices of the target classes
        None: all classes are target classes
    """
    target_tensor = torch.zeros_like(probs) 
    
    if isinstance(target_idx, int):
        target_tensor[:, target_idx] = 1.0
    elif isinstance(target_idx, list):
        for i in target_idx:
            target_tensor[:, i] = 1.0
    elif target_idx == None:
        target_tensor = torch.ones_like(probs)        
    return target_tensor

def compute_object_detection_loss(probs, target):
        # object_detection_loss_target = object_detection_loss_target(probs, target_idx)
        
        # cls_loss = 1.0 * F.mse_loss(prob, object_detection_loss_target, reduction='sum') / (len(logits) + 1)
        
        # total_loss = cls_loss
        # # clear grad
        # adam_opt.zero_grad()
        # total_loss.backward(retain_graph=True)
        
        # manually update
        # self.bx.grad = self.bx.grad / (torch.norm(self.bx.grad,p=2) + 1e-20)
        # self.bx.data = -1.5 * self.bx.grad+ self.bx.data
        
        # with torch.no_grad():
        #     self.bx.data.clamp_(-0.04, 0.04)  
    pass

def load_image_captioning_model(device):
    processor = AutoProcessor.from_pretrained("microsoft/git-base", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
    return model, processor

def mayTheForceBeWithYou(od, od_processor, ic, ic_processor, data, device):
    image_id, image, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(data)
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = od_processor(image, return_tensors="pt")["pixel_values"].to(device)
    img_tensor = utils.denormalize(img_tensor)
    target_size = [img_tensor.shape[2:] for _ in range(1)]
    output = od(img_tensor)
    # Post-process detection results
    _output = od_processor.post_process_object_detection(output, 
                                                        threshold=0.25, 
                                                        target_sizes=target_size)[0]
    
    scores, labels, boxes = _output["scores"], _output["labels"], _output["boxes"]
    
    filtered_boxes = []
    for i, label in enumerate(labels):
        if label == 0 or label == 2:
            filtered_boxes.append(boxes[i].cpu().numpy())
    
    # Merge boxes
    if filtered_boxes:
        # Convert to numpy for easier manipulation
        filtered_boxes = np.array(filtered_boxes)
        
        # Find the bounding coordinates for the merged box
        x_min = np.min(filtered_boxes[:, 0])
        y_min = np.min(filtered_boxes[:, 1])
        x_max = np.max(filtered_boxes[:, 2])
        y_max = np.max(filtered_boxes[:, 3])
        
        # Ensure coordinates are within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        # Crop the image using the merged box
        from PIL import Image
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
    else:
         cropped_image = img_tensor.clone().detach().cpu().numpy()
    

    pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_data = load_ms_coco_dataset(1000)
    
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    od, od_processor = load_rt_detr(0, num_q=1000, device=device)
    ic, ic_processor = load_image_captioning_model(device)
    
    for index, data in tqdm(enumerate(coco_data), total=coco_data.__len__()):
        mayTheForceBeWithYou(od, od_processor, ic, ic_processor, data, device)
        pass
    
    
    
    
    