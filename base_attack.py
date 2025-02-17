from tqdm import tqdm
import torch
import os
import time
import numpy as np
import cv2
import torch.nn.functional as F
from pathlib import Path
import sys
import multiprocessing as mp
import json
import random
import pdb
from typing import Optional, Union, Any
from datasets import load_dataset
from PIL import Image
import requests
from transformers import DetrConfig, AutoImageProcessor, DetrForObjectDetection
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from datetime import datetime


###
import utils

class BaseAttack:
    def __init__(self, 
                 model, 
                 image_processor, 
                 it_num: int,  
                 conf_thres: float = 0.25,  
                 target_idx = None, 
                 output_dir: str = None,  
                 device: Optional[Union[str, torch.device]] = None):
        
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.it_num = it_num # iteration number of the attack
        self.target_idx = target_idx # list of int
        self.best_count = -1 # max count of bbox of the desired class
        self.clean_flag = True
        self.output_dir = output_dir
        self.conf_thres = conf_thres
            
    def generate_bx(self):
        """
            generates the original bx
        """
        bx = np.zeros((self.img_tensor.shape[1], 
                       self.img_tensor.shape[2], 
                       self.img_tensor.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(self.device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        self.bx = bx
        return bx
    
    def init_input(self, img, img_id):
        if img.mode != "RGB":
            img = img.convert("RGB")
        self.img_id = img_id
        self.img_tensor = self.image_processor(img, return_tensors="pt")["pixel_values"].to(self.device)
        self.img_tensor = utils.denormalize(self.img_tensor)
        self.target_size = [self.img_tensor.shape[2:] for _ in range(1)]

    
    def inference(self, input):
        """
            input is the denorm_img_tensor, or (denorm_img_tensor + bx)
        """
        start_time = time.perf_counter()
        self.output = self.model(input)
        end_time = time.perf_counter()
        self.elapsed_time = round((end_time - start_time) * 1000, 2)
        
        self.logits = self.output.logits[0]
        self.prob = F.softmax(self.logits, dim=-1)  
        self.scores, self.labels, self.boxes = self.parse_output()
        self.combined = torch.cat((self.boxes, 
                                   self.scores.unsqueeze(1), 
                                   self.prob[:len(self.scores)]), dim=1)

        if self.clean_flag:
            
            self.cl_scores = self.scores
            self.cl_labels = self.labels
            self.cl_boxes = self.boxes
            self.cl_prob = self.prob
            self.cl_count = len(self.cl_boxes)
            self.cl_infer_t = self.elapsed_time
            self.clean_flag = False
            self.cl_combined = torch.cat((self.cl_boxes, 
                                          self.cl_scores.unsqueeze(1),
                                          self.cl_prob[:len(self.cl_scores)]), dim=1)

        return self.output
                    
    def run_attack(self):
        """
            should be defined in each attack
        """
        pass
            
    def update_bx(self):
        """
            should be defined in each attack
        """
        pass
    
    # def denormalize(self, tensor):
    #     """
    #         Denormalizes a tensor using the provided mean and std.
    #     """
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #     mean = torch.tensor(mean).view(-1, 1, 1).to(self.device)
    #     std = torch.tensor(std).view(-1, 1, 1).to(self.device)
    #     tensor = tensor * std + mean
    #     return tensor
    
    def id2label(self, labels):
        """
            Input: tensor of ids, tensor.dim() == 1
            return a list of label names
        """
        return [self.model.config.id2label[id] for id in labels.tolist()]

                
    
    def parse_output(self, custom_thres=None):
        # target_size = [self.img_tensor.shape[2:] for _ in range(1)]
        if custom_thres == None:
            custom_thres = self.conf_thres
        _output = self.image_processor.post_process_object_detection(self.output, 
                                                                     threshold = custom_thres, 
                                                                     target_sizes = self.target_size)[0]
        # _output = self.move_to_cpu(_output)
        # (top_left_x, top_left_y, bottom_right_x, bottom_right_y) 
        return _output["scores"],  _output["labels"], _output["boxes"]
    
    def move_to_cpu(self, data):
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {key: self.move_to_cpu(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.move_to_cpu(item) for item in data]
        else:
            return data
    
    def logger(self):
        count = 0
        if self.target_idx:
            for i in self.target_idx:
                count = count + (self.labels == i).sum().item()
        else:
            count = len(self.labels)
        if count > self.best_count:
            self.best_count = max(count, self.best_count)
            self.best_labels = self.labels
            self.best_boxes = self.boxes
            self.best_scores = self.scores
            self.best_infer_t = self.elapsed_time
            
    def write_log(self):
        result_dict= {
            "clean_bbox_count": self.cl_count,
            "clean_labels": self.cl_labels.tolist(),
            "clean_scores": self.cl_scores.tolist(),
            "clean_boxes": self.cl_boxes.tolist(),
            "clean_infer_time": self.cl_infer_t,
            "corrupted_bbox_count": self.best_count,
            "corrupted_labels": self.best_labels.tolist(),
            "corrupted_scores": self.best_scores.tolist(),
            "corrupted_boxes": self.best_boxes.tolist(),
            "corrupted_infer_time": self.best_infer_t
        }

        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"img_id_{str(self.img_id)}.json")
        with open(file_path, 'w', encoding="utf-8") as json_file:
            json.dump(result_dict, json_file, indent=4)
        
    
    def parse_example(self, example):
        image_id = example["image_id"]
        image = example["image"]
        width = example["width"]
        height = example["height"]
        bbox_id = example["objects"]["bbox_id"]
        category = example["objects"]["category"]
        bbox = example["objects"]["bbox"]
        area = example["objects"]["area"]
        
        return image_id, image, width, height, bbox_id, category, bbox, area
    


if __name__ == "__main__":
    pass