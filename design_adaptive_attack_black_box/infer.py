from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

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
import torch.nn as nn
import torchvision

import sys
from pathlib import Path

YOLOV5_FILE = Path(f"../model/yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add ROOT to PATH
from models.common import DetectMultiBackend
from utils.general import Profile, non_max_suppression

from PIL import Image
import logging

# from adaptive_attack import add_gaussian_noise, add_spatial_smoothing

def create_logger(module, filename, level):
    # Create a formatter for the logger, setting the format of the log with time, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a logger named 'logger_{module}'
    logger = logging.getLogger(f'logger_{module}')
    logger.setLevel(level)     # Set the log level for the logger
    
    # Create a file handler, setting the file to write the logs, level, and formatter
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(level)         # Set the log level for the file handler
    fh.setFormatter(formatter) # Set the formatter for the file handler
    
    # Add the file handler to the logger
    logger.addHandler(fh)
    
    return logger

def add_gaussian_noise(tensor, mean=0.0, std=5.0):
    # print(f"tensor: {tensor}")
    
    # global global_std
    # global_std += 1
    # global_std = global_std % 100 + 1
    # print(f"global_std = {global_std}")
    # std = global_std
    
    noise = torch.randn_like(tensor) * std + mean
    # print(f"noise: {noise}")
    noise /= 255.0
    # print(f"noise: {noise}")
    # time.sleep(5)
    noisy_tensor = tensor + noise
    return noisy_tensor


def add_spatial_smoothing(images, kernel_size=3):
# def add_spatial_smoothing(images):
    """
    对输入图像进行空间平滑处理
    参数:
    - images: 输入图像的张量 (N, C, H, W)
    - kernel_size: 平滑滤波器的大小，默认为3
    
    返回值:
    - 平滑后的图像张量
    """
    # global global_kernel_size
    # global_kernel_size += 1
    # global_kernel_size = global_kernel_size % 10 + 1
    # print(f"global_kernel_size = {global_kernel_size}")
    # kernel_size = global_kernel_size
    
    # 构建一个均值滤波器核
    padding = kernel_size // 2
    smoothing_filter = torch.ones((images.shape[1], 1, kernel_size, kernel_size), device=images.device) / (kernel_size * kernel_size)
    
    # 对每个通道进行卷积操作实现空间平滑
    smoothed_images = F.conv2d(images, smoothing_filter, padding=padding, groups=images.shape[1])
    
    return smoothed_images

def infer(image_path):
    processed_image = cv2.imread(image_path)
    
    image = cv2.imread(image_path)
    
    # img_size = (1082, 602)
    # processed_image = cv2.resize(processed_image, img_size)
    # image = cv2.resize(image, img_size)
    
    # print(f"image.shape = {image.shape}") # (608, 1088, 3)
    # print(f"image = {image}")

    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device).float()
    image /= 255.0
    
    if len(image.shape) == 3:
        image = image[None]
    
    # tensor_type = torch.cuda.FloatTensor
    # image_tensor = image.type(tensor_type)
    # image_tensor = image.to(device)
    
    image_tensor = image
    
    # print(f"image_tensor = {image_tensor}")
    
    # image_tensor = add_spatial_smoothing(image_tensor)
    # image_tensor = add_gaussian_noise(image_tensor)
    
    outputs = model(image_tensor)
    
    # print(f"outputs = {outputs}")
    
    outputs = outputs[0].unsqueeze(0)
    
    # scores = outputs[..., index] * outputs[..., 4]
    # scores = scores[scores > 0.25]
    # print(f"len(scores) = {len(scores)}")
    # objects_num_before_nms = len(scores) # 实际上是 {attack_object} number before NMS
    
    conf_thres = 0.25 # 0.25  # confidence threshold
    iou_thres = 0.45  # 0.45  # NMS IOU threshold
    max_det = 100000    # maximum detections per image
    
    xc = outputs[..., 4] > 0
    x = outputs[0][xc[0]]
    x[:, 5:] *= x[:, 4:5]
    max_scores = x[:, 5:].max(dim=-1).values
    objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
    objects_num_after_nms = 0
    person_num_after_nms = 0
    car_num_after_nms = 0
    
    outputs = non_max_suppression(outputs, conf_thres, iou_thres, max_det=max_det)
    
    for i, det in enumerate(outputs): # detections per image
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f"{names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence}" # f"{confidence:.2f}"
                box = [round(float(i), 2) for i in xyxy]
                # print(f"Detected {label} with confidence {confidence_str} at location {box}")
                
                if label == "person":
                    person_num_after_nms += 1
                elif label == "car":
                    car_num_after_nms += 1
                
                # print(f"Detected {label} with confidence {confidence_str} at location {box}")
                # time.sleep(5)
                processed_image = cv2.rectangle(processed_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2) 
                
            objects_num_after_nms = len(det)
        # print(f"There are {len(det)} objects detected in this image.")
    
    # Draw the processed image
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, processed_image)
    
    # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
    print(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, car_num_after_nms = {car_num_after_nms}")
    return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms


def dir_process(dir_path):
    image_list = []
    image_name_list = os.listdir(dir_path)
    image_name_list.sort()
    # print(f"image_name_list = {image_name_list}")
    for image_name in image_name_list:
        if image_name.endswith(".png"):
            image_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_path)
            # print(f"image.shape = {image.shape}") # (608, 1088, 3)
            image_list.append(image)

    return image_list, image_name_list

if __name__ == "__main__":
    # weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    weights = "./model/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    device = torch.device('cuda:1')
    model = DetectMultiBackend(weights=weights, device=device)
    names = model.names
    print(f"names = {names}")
    
    attack_method = f"adaptive_attack_original"
    
    logger_dir = "infer/log"
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)
    logger_path = f"{logger_dir}/{attack_method}.log"
    logger = create_logger(f"{attack_method}", logger_path, logging.INFO)

    input_dir = f"infer/input/{attack_method}"
    output_dir = f"infer/output/{attack_method}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_size = (608, 1088)
    image_list, image_name_list = dir_process(input_dir)
    
    # TODO: 加入低扰动仍然无法检测到之前的目标，说明 attack 改的像素点太多了
    for image_name in image_name_list:
        image_path = os.path.join(input_dir, image_name)
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms = infer(image_path)
        logger.info(f"{image_name} {objects_num_before_nms} {objects_num_after_nms} {person_num_after_nms} {car_num_after_nms}")
