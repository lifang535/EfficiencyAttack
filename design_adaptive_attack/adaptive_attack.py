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


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

epoch_id = 0 # lifang535: for adaptive attack
def run_attack(outputs,bx, strategy, max_tracker_num, adam_opt):
    global epoch_id
    
    outputs = outputs[0][0] # lifang535 add
    
    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    scores = outputs[:,index] * outputs[:,4]
    height = outputs[:,2]
    width = outputs[:,3]
    
    # print(f"scores.shape = {scores.shape}, height.shape = {height.shape}, width.shape = {width.shape}")

    # # lifang535: 用于破坏 threshold 的防御
    # scores_1 = scores[height * width >= 900]
    # scores_2 = scores[height * width < 900]
    # scores = torch.cat((scores_1, torch.zeros_like(scores_2)), dim=0)
    # scores = threshold_fitting(scores, height, width)

    # lifang535: sel_dets 和 SlowTrack 的 feature matching 有关，由于我们不考虑连续图片之间的关系，这里全部选中
    sel_dets = scores 
    sel_height = height
    sel_width = width
    sel_aaa = (sel_width/1088) * (sel_height/608)
    
    targets = torch.ones_like(sel_dets)
    loss_score = 1.0*(F.mse_loss(sel_dets, targets, reduction='sum'))
    
    loss_area = 100*torch.sum(sel_aaa) # lifang535: 相较于 stra_attack，这里的 loss4 是对所有的 box 的面积求和
    
    loss_bx = 40*torch.norm(bx, p=2)
    
    # lifang535: 动态损失函数
    if epoch_id <= 50:
        loss = loss_score
    else:
        loss = 10*loss_score+loss_area+10*loss_bx
    
    loss.requires_grad_(True)
    adam_opt.zero_grad()
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    count = (scores > 0.25).sum()
    print('loss',loss.item(),'loss_score',loss_score.item(),'loss_area',loss_area.item(),'loss_bx',loss_bx.item(),'count:',count.item())
    return bx

# lifang535: 用于破坏 gaussian_noise 的防御
global_std = 1.0
def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    # print(f"tensor: {tensor}")
    
    global global_std
    global_std += 1
    global_std = global_std % 100 + 1
    print(f"global_std = {global_std}")
    std = global_std
    
    noise = torch.randn_like(tensor) * std + mean
    # print(f"noise: {noise}")
    noise /= 255.0
    # print(f"noise: {noise}")
    # time.sleep(5)
    noisy_tensor = tensor + noise
    return noisy_tensor

# lifang535: 用于破坏 spatial_smoothing 的防御
global_kernel_size = 1
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
    global global_kernel_size
    global_kernel_size += 1
    global_kernel_size = global_kernel_size % 10 + 1
    print(f"global_kernel_size = {global_kernel_size}")
    kernel_size = global_kernel_size
    
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
        self, image_list, image_name_list, img_size): # args, dataloader, img_size, confthre, nmsthre, num_classes):
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
        self.img_size = img_size
        
        self.dataloader = None
        self.confthre = 0.25
        self.nmsthre = 0.45
        self.num_classes = None
        self.args = None

    def evaluate(
        self,
        imgs,
        image_name,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
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
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        frame_id = 0
        total_l1 = 0
        total_l2 = 0
        strategy = 1
        max_tracker_num = int(8)
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        # for cur_iter, (imgs,path) in enumerate(
        #     progress_bar(self.dataloader)
        #     ):

        # lifang535: strategy 和 SlowTrack 的 feature matching 有关，由于我们不考虑连续图片之间的关系，没有使用
        print('strategy:',strategy)
        
        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        adam_opt = Adam([bx], lr=learning_rate, amsgrad=True)
        imgs = imgs.type(tensor_type)
        imgs = imgs.to(device)
        #(1,23625,6)
        
        for iter in tqdm(range(epochs)):
            epoch_id = iter # lifang535: for adaptive attack
            
            added_imgs = imgs+bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            added_imgs.clamp_(min=0, max=1)
            input_imgs = (added_imgs - rgb_means)/std
            if half:
                input_imgs = input_imgs.half()
            # outputs = model(input_imgs) # lifang535: 这个是 SlowTrack 原始代码，但是会影响攻击效果
            # outputs = model(added_imgs) # lifang535: 直接用 added_imgs 攻击效果较好

            # noise_imgs = add_gaussian_noise(added_imgs) # lifang535: 用来破坏 gaussian_noise 的防御
            # noise_imgs = add_spatial_smoothing(added_imgs) # lifang535: 用来破坏 spatial_smoothing 的防御
            noise_imgs = added_imgs
            outputs = model(noise_imgs)
            
            bx = run_attack(outputs,bx, strategy, max_tracker_num, adam_opt)
        if strategy == max_tracker_num-1:
            strategy = 0
        else:
            strategy += 1
        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        
        
        input_path = f"{input_dir}/{image_name}"
        output_path = f"{output_dir}/{image_name}"
        cv2.imwrite(output_path, added_blob) # lifang535: 这个 attack 效果似乎不受小数位损失影响
        
        print(f"[AdaptiveAttack.evaluate] Saved image to {output_path}")
        objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms = infer(input_path)
        _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _target_num_after_nms = infer(output_path)
        
        # logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, {attack_object}_num_after_nms: {target_num_after_nms}, _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _{attack_object}_num_after_nms: {_target_num_after_nms}")
        logger.info(f"{objects_num_before_nms} {objects_num_after_nms} {person_num_after_nms} {target_num_after_nms} {_objects_num_before_nms} {_objects_num_after_nms} {_person_num_after_nms} {_target_num_after_nms}")
        
        print(l1_norm.item(),l2_norm.item())
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        print(mean_l1.item(),mean_l2.item())
        del bx
        del adam_opt
        del outputs
        del imgs

        return mean_l1, mean_l2

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xywh2xyxy(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def run(self):
        """
        Run the evaluation.
        """
        for image, image_name in zip(self.image_list, self.image_name_list):
            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(device).float()
            image /= 255.0

            if len(image.shape) == 3:
                image = image[None]

            # print(f"image.shape = {image.shape}")
            
            mean_l1, mean_l2 = self.evaluate(image, image_name)


def infer(image_path):
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
    weights = "./model/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    device = torch.device('cuda:3')
    model = DetectMultiBackend(weights=weights, device=device)
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
    
    attack_object_key = 0 # 0: person, 2: car
    attack_method = f"adaptive_attack_original"
    
    for attack_object_key in [0]:
        # TODO: 一些 label 一旦出现一个，就会出现再出现很多，或许可以先注入，或者增大迭代次数
        # attack_object_key = 0
        attack_object = names[attack_object_key]
        index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

        epochs = 200
        learning_rate = 0.01 #0.07
        
        logger_dir = f"{attack_method}/log/epochs_{epochs}"
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        logger_path = f"{logger_dir}/{attack_object_key}_{attack_object}.log"
        
        logger = create_logger(f"{attack_method}_{attack_object}_epochs_{epochs}", logger_path, logging.INFO)

        input_dir = "original"
        output_dir = f"{attack_method}/output/epochs_{epochs}/{attack_object_key}_{attack_object}"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        img_size = (608, 1088)
        image_list, image_name_list = dir_process(input_dir)
        
        aa = AdaptiveAttack(
            image_list=image_list,
            image_name_list=image_name_list,
            img_size=img_size,
        )
        
        aa.run()
