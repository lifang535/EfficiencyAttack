import random
from datasets import load_dataset
import CONSTANTS
import torch
from transformers import DetrConfig, AutoImageProcessor, DetrForObjectDetection

from PIL import Image
import requests
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2
import logging
from PIL import Image, ImageDraw
from torchvision.utils import save_image
import util
import pdb
random.seed(42)

if __name__ == "__main__":
    url = "https://farm5.staticflickr.com/4116/4827719363_31f75f0c8f_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:1')
    
    configuration =  DetrConfig()
    configuration.num_queries = 100
    print(configuration)
    
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    # model = DetrForObjectDetection(configuration).to(device)
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    # model.config.num_queries = 1000

    model.eval()
    input = image_processor(images=image, return_tensors="pt").to(device)
    img_tensor = input["pixel_values"]
    img_tensor = util.denormalize(img_tensor)
    target_size = [img_tensor.shape[2:] for _ in range(1)]
    
    result = model(img_tensor)
    # Apply softmax to compute probabilities
    logits = result.logits[0]
    # probabilities = F.softmax(logits, dim=-1)  
    probabilities = F.sigmoid(logits)  

    # Get the highest probability and corresponding class for each query
    max_probs, predicted_classes = probabilities.max(dim=-1)  # Shape: [100]

    # Filter scores greater than 0.25
    threshold = 0.25
    high_conf_indices = (max_probs > threshold).nonzero(as_tuple=True)

    # Get the high-confidence queries
    filtered_probs = max_probs[high_conf_indices]
    filtered_classes = predicted_classes[high_conf_indices]
    filtered_indices = high_conf_indices[0]
    # Query indices

    # Print results
    for i in range(len(filtered_probs)):
        print(f"Query Index: {filtered_indices[i]}, Class: {filtered_classes[i]}, Confidence: {filtered_probs[i].item()}")
    
    output = image_processor.post_process_object_detection(result, 
                                                           threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                           target_sizes = target_size)[0]
    
    height, width = target_size[0][0], target_size[0][1]
    scores, labels, boxes = util.parse_prediction(output)
    # normed_detr_boxes = util.scale_boxes(boxes, height, width)
    yolo_boxes = util.xxyy2xywh(boxes)
    
    pdb.set_trace()
    # result.logits.shape
    # print(model.config)
    # print(dir(configuration))
    
    # TODO:
    # targeted label
    # box area
    # best label
    # balance label
    
    # do not do eng work
    # loss function 
    
    # find bottle neck (label) is also a part of our approach
    
    # TODO:
    # world model for robotic arm
    # env info 
    # Mixture of Expert (efficiency attack) -> ravishka
    # explore security topics