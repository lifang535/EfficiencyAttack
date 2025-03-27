import io
import os
import cv2
import sys
import csv
import json
import time
import logging
import tempfile
import itertools
import contextlib
import torchvision

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from math import sin, cos, pi
from collections import defaultdict

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

# yolov5
YOLOV5_FILE = Path(f"../model/yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add ROOT to PATH
from models.common import DetectMultiBackend
from utils.general import Profile, non_max_suppression as non_max_suppression_yolov5

# yolov8
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.ops import non_max_suppression as non_max_suppression_yolov8

from utils_temp import create_logger

def run_attack(outputs_list, bx, adam_opt, epoch_id):
    loss_sum = 0
    for outputs in outputs_list:
        outputs = outputs[0][0] # lifang535 add

        scores = outputs[:,index] * outputs[:,4]
        height = outputs[:,2]
        width = outputs[:,3]
        
        # print(f"scores.shape = {scores.shape}, height.shape = {height.shape}, width.shape = {width.shape}")

        # # lifang535: 用于破坏 threshold 的防御（控制输出 bounding box 大于某个 threshold）
        # scores_1 = scores[height * width >= 900]
        # scores_2 = scores[height * width < 900]
        # scores = torch.cat((scores_1, torch.zeros_like(scores_2)), dim=0)
        # scores = threshold_fitting(scores, height, width)

        # lifang535: sel_dets 和 SlowTrack 的 feature matching 有关，由于我们不考虑连续图片之间的关系，这里全部选中
        sel_dets = scores 
        sel_height = height
        sel_width = width
        sel_aaa = (sel_width/640) * (sel_height/640)
        
        targets = torch.ones_like(sel_dets)
        loss_score = 1.0*(F.mse_loss(sel_dets, targets, reduction='sum'))
        loss_score /= len(sel_dets) # 归一化 loss_score
        
        loss_area = 100*torch.sum(sel_aaa) # lifang535: 相较于 stra_attack，这里的 loss4 是对所有的 box 的面积求和
        loss_area /= len(sel_dets) # 归一化 loss_area
        
        # loss_bx = torch.norm(bx, p=2) # l2 norm
        # loss_bx = loss_bx / 50.0
        
        loss_l1 = torch.norm(bx, p=1) # l1 norm
        loss_l1 = loss_l1 / (bx.shape[3] * bx.shape[2])
        loss_l2 = torch.norm(bx, p=2) # l2 norm
        loss_l2 = loss_l2 / 50.0
        loss_linf = torch.norm(bx, p=float('inf')) # l_inf norm
        loss_linf = loss_linf / 50
        # print(f"loss_l1 = {loss_l1.item()}, loss_l2 = {loss_l2.item()}, loss_linf = {loss_linf.item()}")
        # [] 0.0 0.0 0.0
        loss_bx = loss_l1 + loss_l2 # + loss_linf
        
        
        alpha2 = 1 - cos(min(epoch_id / (epochs / 2) * pi, pi / 2))
        alpha3 = 1 - cos(min(epoch_id / (epochs / 2) * pi, pi / 2))
        alpha1 = 3 - alpha2 - alpha3
        
        loss = alpha1 * loss_score + alpha2 * loss_area + alpha3 * loss_bx
        
        loss_sum += loss
        
    loss_sum.requires_grad_(True)
    adam_opt.zero_grad()
    loss_sum.backward(retain_graph=True)
    
    adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    
    # 限制 bx 的范围
    epsilon = 10
    bx.data = torch.clamp(bx.data, -epsilon * 1 / 255, epsilon * 1 / 255)
    
    count = (scores > 0.25).sum()
    # print('loss',loss.item(),'loss_score',loss_score.item(),'loss_area',loss_area.item(),'loss_bx',loss_bx.item(),'count:',count.item()) # lifang535 delete
    # print('loss',loss.item(),'loss_score',loss_score.item(),'loss_area',loss_area.item(),'loss_bx',loss_bx.item(),'loss_l1',loss_l1.item(),'loss_l2',loss_l2.item(),'loss_linf',loss_linf.item(),'count:',count.item())
    return bx

# lifang535: 用于破坏 gaussian_noise 的防御
# global_std = 20.0
def add_gaussian_noise(images, mean=0.0, std=1.0):
    noise = torch.randn_like(images) * std + mean
    noise /= 255.0
    noisy_images = images + noise
    return noisy_images

# lifang535: 用于破坏 spatial_smoothing 的防御
# global_kernel_size = 1
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

# lifang535: 用于破坏 threshold 的防御
def threshold_fitting(scores, height, width, area_thres=900, height_max=1088, width_max=608):
    # 把 area 小于 area_thres 的 box 的 score 设为 0
    scores[height * width < area_thres] = 0
    return scores

class AdaptiveAttack:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, image_list, image_name_list, image_size, image_process_func=None):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.image_list = image_list
        self.image_name_list = image_name_list
        self.image_size = image_size
        self.image_process_func = image_process_func
        
        self.dataloader = None
        self.confthre = 0.25
        self.nmsthre = 0.45
        self.num_classes = None
        self.args = None

    def evaluate(
        self,
        imgs,
        image_name,
        image_process_func=None,
    ):
        global model, names, device
        global epoch_id # lifang535: for new attack
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        tensor_type = torch.cuda.FloatTensor
        model = model.eval()

        strategy = 1 # strategy 和 SlowTrack 的 feature matching 有关，由于我们不考虑连续图片之间的关系，没有使用
        max_tracker_num = int(8)
        
        rgb_means = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)

        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        adam_opt = Adam([bx], lr=learning_rate, amsgrad=True)
        imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)

        for iter in tqdm(range(epochs)):
            added_imgs = imgs + bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            
            added_imgs.clamp_(min=0, max=1)
            input_imgs = (added_imgs - rgb_means) / std
            
            # outputs = model(input_imgs) # lifang535: 这个是 SlowTrack 原始代码，但是会影响攻击效果
            # outputs = model(added_imgs) # lifang535: 直接用 added_imgs 攻击效果较好

            if image_process_func is None:
                processed_imgs = added_imgs
            elif image_process_func == "gaussian_noise":
                processed_imgs = add_gaussian_noise(images=added_imgs,
                                                    mean=0.0,
                                                    std=gaussian_noise_std) # 为图片添加高斯噪声
            elif image_process_func == "spatial_smoothing":
                processed_imgs = add_spatial_smoothing(images=added_imgs,
                                                       kernel_size=spatial_smoothing_kernel_size) # 为图片添加空间平滑
            else:
                raise ValueError(f"Unknown image_process_func: {image_process_func}")
            
            output_1 = model_1(processed_imgs)
            output_2 = model_2(processed_imgs)
            output_3 = model_3(processed_imgs)
            
            outputs_list = [output_1, output_2, output_3]

            bx = run_attack(outputs_list=outputs_list,
                            bx=bx,
                            adam_opt=adam_opt,
                            epoch_id=iter)

        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        
        
        input_path = f"{input_dir}/{image_name}"
        output_path = f"{output_dir}/{image_name}"
        cv2.imwrite(output_path, added_blob) # lifang535: 这个 attack 效果似乎不受小数位损失影响
        
        print(f"[AdaptiveAttack.evaluate] Saved image to {output_path}")
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms = infer(input_path, model_black)
        _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _target_num_after_nms = infer(output_path, model_black)
        
        # model_1
        print(f"model_1")
        _, _, _, _ = infer(output_path, model_1)
        # model_2
        print(f"model_2")
        _, _, _, _ = infer(output_path, model_2)
        # model_3
        print(f"model_3")
        _, _, _, _ = infer(output_path, model_3)
        # model_black
        print(f"model_black")
        _, _, _, _ = infer(output_path, model_black)
        # model_yolov8
        print(f"model_yolov8")
        _, _, _, _ = infer_yolov8(output_path, model_yolov8)
        
        # 计算修改像素值大小
        diff_avg, diff_max, diff_norm2 = compare_images(input_path, output_path)
        print(f"diff_avg = {diff_avg}, diff_max = {diff_max}, diff_norm2 = {diff_norm2}")

        # logger.info(f"{objects_num_before_nms} {objects_num_after_nms} {person_num_after_nms} {target_num_after_nms} {_objects_num_before_nms} {_objects_num_after_nms} {_person_num_after_nms} {_target_num_after_nms}")
        
        attack_data = [image_name, objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms, _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _target_num_after_nms, diff_avg, diff_max, diff_norm2]
        
        with open(csv_path, "a") as file:
            csv_writer = csv.writer(file)
            # 写入 attack_data 列表
            csv_writer.writerow(attack_data)
        
        print(l1_norm.item(), l2_norm.item())

        del bx
        del adam_opt
        del outputs_list
        del imgs

        return l1_norm, l2_norm

    def run(self):
        """
        Run the evaluation.
        """
        image_id = 0
        total_l1 = 0
        total_l2 = 0
        
        for image, image_name in zip(self.image_list, self.image_name_list):
            image_id += 1
            
            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(device).float()
            image /= 255.0

            if len(image.shape) == 3:
                image = image[None]

            # print(f"image.shape = {image.shape}")
            
            l1_norm, l2_norm = self.evaluate(image, image_name, self.image_process_func)
            
            total_l1 += l1_norm
            total_l2 += l2_norm
            
        mean_l1 = total_l1 / image_id
        mean_l2 = total_l2 / image_id
        
        print(f"mean_l1 = {mean_l1}, mean_l2 = {mean_l2}")
        return mean_l1, mean_l2


def infer(image_path, model):
    image = cv2.imread(image_path)
    image = image.transpose((2, 0, 1))[::-1]
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device).float()
    image /= 255.0
    if len(image.shape) == 3:
        image = image[None]
        
    image_tensor = image
    outputs = model(image_tensor)
    
    outputs = outputs[0].unsqueeze(0)
    
    # scores = outputs[..., index] * outputs[..., 4]
    # scores = scores[scores > 0.25]
    # print(f"len(scores) = {len(scores)}")
    # objects_num_before_nms = len(scores) # 实际上是 {attack_object} number before NMS
    
    conf_thres = 0.25 # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 100000  # maximum detections per image
    
    xc = outputs[..., 4] > 0
    x = outputs[0][xc[0]]
    x[:, 5:] *= x[:, 4:5]
    max_scores = x[:, 5:].max(dim=-1).values
    objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
    objects_num_after_nms = 0
    person_num_after_nms = 0
    target_num_after_nms = 0
    
    outputs = non_max_suppression_yolov5(outputs, conf_thres, iou_thres, max_det=max_det)
        
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
                if label == attack_object:
                    target_num_after_nms += 1
            objects_num_after_nms = len(det)
        # print(f"There are {len(det)} objects detected in this image.")
    
    # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms
    print(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, {attack_object}_num_after_nms = {target_num_after_nms}")
    return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms

def compare_images(input_path, output_path):
    input_image = cv2.imread(input_path)
    output_image = cv2.imread(output_path)
    
    # print(f"input_image.shape = {input_image.shape}, output_image.shape = {output_image.shape}")
    
    # 计算平均修改像素值
    diff_avg = cv2.absdiff(input_image, output_image)
    diff_avg = diff_avg.astype(np.float32)
    diff_avg = diff_avg.sum() / (input_image.shape[0] * input_image.shape[1] * input_image.shape[2])
    
    # 计算最大修改像素值
    diff_max = cv2.absdiff(input_image, output_image)
    diff_max = diff_max.astype(np.float32)
    diff_max = diff_max.max()
    
    # 计算 L2 范数
    # diff = cv2.absdiff(input_image, output_image)
    # diff = diff.astype(np.float32)
    # diff_norm2 = np.linalg.norm(diff)
    diff_norm2 = np.mean((input_image.astype(np.float32) - output_image.astype(np.float32)) ** 2)
    diff_norm2 = np.sqrt(diff_norm2)
    
    return diff_avg, diff_max, diff_norm2

def infer_yolov8(image_path, model):
    image = cv2.imread(image_path)

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
    
    outputs = model(image_tensor)
    
    outputs_copy = [outputs[0].clone()] # lifang535 add
    
    # print(f"outputs = {outputs}")
    
    outputs = outputs[0].unsqueeze(0)
    
    # (1, 1, 84, 13566) -> (1, 1, 13566, 84)
    outputs = outputs.permute(0, 1, 3, 2)
    
    # scores = outputs[..., index] * outputs[..., 4]
    # scores = scores[scores > 0.25]
    # print(f"len(scores) = {len(scores)}")
    # objects_num_before_nms = len(scores) # 实际上是 {attack_object} number before NMS
    
    conf_thres = 0.25 # 0.25  # confidence threshold
    iou_thres = 0.45  # 0.45  # NMS IOU threshold
    max_det = 100000  # maximum detections per image
    
    xc = outputs[..., 4] > 0
    x = outputs[0][xc[0]]
    # x[:, 5:] *= x[:, 4:5]
    max_scores = x[:, 4:].max(dim=-1).values
    objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
    objects_num_after_nms = 0
    person_num_after_nms = 0
    target_num_after_nms = 0
    
    # outputs = non_max_suppression_yolov5(outputs, conf_thres, iou_thres, max_det=max_det)
    # outputs = non_max_suppression_yolov5(outputs_copy, conf_thres, iou_thres, max_det=max_det)
    outputs = non_max_suppression_yolov8(outputs_copy, conf_thres, iou_thres, max_det=max_det)
    
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
                if label == attack_object:
                    target_num_after_nms += 1
            objects_num_after_nms = len(det)
        # print(f"There are {len(det)} objects detected in this image.")
    
    # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms
    print(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, {attack_object}_num_after_nms = {target_num_after_nms}")
    return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms


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
            
            # # 将图片大小扩充为 608x1088，扩充黑色像素点 # lifang535: for animal test
            # image = cv2.copyMakeBorder(image, 0, 608 - image.shape[0], 0, 1088 - image.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # # 保存为 {name}_resized.png
            # cv2.imwrite(f"{dir_path}/{image_name}_resized.png", image)
            # image = cv2.imread(f"{dir_path}/{image_name}_resized.png")
            # image_name_list[image_name_list.index(image_name)] = f"{image_name}_resized.png"
            
            image_list.append(image)

    return image_list, image_name_list


if __name__ == "__main__":
    weights = "../model/yolov5/yolov5x.pt" # yolov5n.pt yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    
    weights_n = "../model/yolov5/yolov5n.pt"
    weights_s = "../model/yolov5/yolov5s.pt"
    weights_m = "../model/yolov5/yolov5m.pt"
    weights_l = "../model/yolov5/yolov5l.pt"
    
    weights_yolov8n = "../model/ultralytics/yolov8n.pt"
    
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights=weights_n, device=device)
    
    model_1 = DetectMultiBackend(weights=weights_n, device=device)
    model_2 = DetectMultiBackend(weights=weights_s, device=device)
    model_3 = DetectMultiBackend(weights=weights_m, device=device)
    
    model_black = DetectMultiBackend(weights=weights_l, device=device)
    
    model_yolov8 = AutoBackend(weights=weights_yolov8n, device=device)
    
    names = model.names
    print(f"names = {names}")
    '''
    names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    '''
    # 0 2 23 43 68
    attack_object_key = 0 # 0: person, 2: car
    attack_method = f"adaptive_attack"
    image_process_func = "gaussian_noise" # None "gaussian_noise" "spatial_smoothing"
    gaussian_noise_std = 20.0
    spatial_smoothing_kernel_size = 3
    
    # TODO: 一些 label 一旦出现一个，就会出现再出现很多，或许可以先注入，或者增大迭代次数
    attack_object = names[attack_object_key]
    index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

    epochs = 400
    # learning_rate = 0.01 #0.07 # lr 过大，曲线震荡，lr 过小，收敛慢
    learning_rate = 0.01 #0.07

    # logger_dir = f"../data/adversarial_log/{attack_method}/epochs_{epochs}"
    # if not os.path.exists(logger_dir):
    #     os.makedirs(logger_dir)
    # logger_path = f"{logger_dir}/{attack_object_key}_{attack_object}.log"
    # logger = create_logger(f"{attack_method}_{attack_object}_epochs_{epochs}", logger_path, logging.INFO)
    
    csv_dir = f"../data/adversarial_csv/{attack_method}/epochs_{epochs}"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = f"{csv_dir}/{attack_object_key}_{attack_object}.csv"
    # 创建 csv 文件
    head_row = ["image_name", "original_object_number", "original_box_number", "original_person_number", "original_target_number", "adversarial_object_number", "adversarial_box_number", "adversarial_person_number", "adversarial_target_number", "diff_avg", "diff_max", "diff_norm2"]
    with open(csv_path, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(head_row)

    input_dir = "../data/original_image"
    output_dir = f"../data/adversarial_image/{attack_method}/epochs_{epochs}/{attack_object_key}_{attack_object}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list, image_name_list = dir_process(input_dir)
    image_size = image_list[0].shape[:2]
    print(f"image_size = {image_size}")
    
    aa = AdaptiveAttack(
        image_list=image_list,
        image_name_list=image_name_list,
        image_size=image_size,
        image_process_func=image_process_func
    )
    
    aa.run()
