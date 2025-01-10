from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import math

import torch

# from yolox.utils import (
#     gather,
#     is_main_process,
#     postprocess,
#     synchronize,
#     time_synchronized,
#     xyxy2xywh
# )
import CONSTANTS
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
# from models.common import DetectMultiBackend
# from utils.general import Profile, non_max_suppression

from PIL import Image
from torchvision import transforms
sys.path.append("../captioning")
from ms_captioning import MSCaptioning
# from detr import target_cls_idx as TARGET_CLS_IDX
# from detr import if_pipeline as IF_PIPELINE
from detr import args
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

epoch_id = 0 # lifang535: for new attack

def threshold_fitting(scores, height, width, area_thres=900, height_max=1088, width_max=608):
    # 把 area 小于 area_thres 的 box 的 score 设为 0，height 和 width 设为 height_max 和 width_max
    # scores.shape = torch.Size([40698]), height.shape = torch.Size([40698]), width.shape = torch.Size([40698])
    for i in range(len(scores)):
        if height[i] * width[i] < area_thres:
            scores[i] = 0
            height[i] = height_max
            width[i] = width_max
    return scores, height, width
    


#TODO：改coefficient
#TODO：justify 设计

    # loss = loss3 * 100 + loss2
    # loss.requires_grad_(True)
    # loss.backward(retain_graph=True)
    # bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    # bx.data = torch.clamp(-3.5 * mask * bx.grad+ bx.data, min=-0.2, max=0.2)
    # count = (scores > 0.9).sum()
    
    if epoch_id <= 50:  # lifang535: for new attack
        if count < CONSTANTS.MAX_COUNT-10:
            loss = loss3 * 100 + loss2
            # loss = loss3 # lifang535 add # lifang535: for new attack
            # loss = loss3+2*(10000-loss2)
        else:
            loss = loss3 # lifang535 add # lifang535: for new attack
        loss.requires_grad_(True)
        adam_opt.zero_grad()
        loss.backward(retain_graph=True)
        bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
        # bx.data = -3.5 * mask * bx.grad+ bx.data
        bx.data = torch.clamp(-1.5 * bx.grad + bx.data, min=-0.2, max=0.2)
        count = (scores > 0.9).sum()
    else: 
        if not good:# lifang535: for new attack
            loss = factor_l4 * loss4 + factor_l3 * loss3 + factor_l2 *loss2 
        else:
            loss = loss2 + loss3 + loss4# lifang535: for new attack
        
        loss.requires_grad_(True)
        adam_opt.zero_grad()
        loss.backward(retain_graph=True)
        
        bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
        bx.data = -1.5 * bx.grad+ bx.data
        count = (scores > 0.25).sum()
        
    
    tqdm.write(f"loss: {loss:.4f} loss_1: {loss1:.4f} loss_2: {loss2:.4f} loss_3: {loss3:.4f} loss_4: {loss4:.4f} good: {good}")

    return bx, count          

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


def old_run_attack(outputs,bx, strategy, max_tracker_num, adam_opt):
    global epoch_id
    
    outputs = outputs[0][0] # lifang535 add
    
    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    scores = outputs[:,index] * outputs[:,4]
    height = outputs[:,2]
    width = outputs[:,3]

    sel_scores_b = scores[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_scores_m = scores[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_scores_s = scores[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]
    sel_height_b = height[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_height_m = height[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_height_s = height[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]
    sel_width_b = width[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_width_m = width[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_width_s = width[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]


    sel_dets = torch.cat((sel_scores_b, sel_scores_m, sel_scores_s), dim=0)
    sel_height = torch.cat((sel_height_b, sel_height_m, sel_height_s), dim=0)
    sel_width = torch.cat((sel_width_b, sel_width_m, sel_width_s), dim=0)
    sel_aaa = (sel_width/640) * (sel_height/640)
    
    # print(f"scores.shape = {scores.shape}, height.shape = {height.shape}, width.shape = {width.shape}")
    # print(f"sel_dets.shape = {sel_dets.shape}, sel_height.shape = {sel_height.shape}, sel_width.shape = {sel_width.shape}, sel_aaa.shape = {sel_aaa.shape}")
    # time.sleep(5)
    
    loss4 = 100*torch.sum(sel_aaa) # lifang535: 相较于 stra_attack，这里的 loss4 是对所有的 box 的面积求和
    targets = torch.ones_like(sel_dets)
    loss1 = 10*(F.mse_loss(sel_dets, targets, reduction='sum')) # lifang535: 和 feature matching 有关
    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores) # lifang535 remove
    # targets = torch.zeros_like(scores) # lifang535 add
    loss3 = 1.0*(F.mse_loss(scores, targets, reduction='sum'))
    # loss = loss1+loss4#+loss3#+loss2 # lifang535 remove
    # loss = loss1+loss4+loss3+loss2 # lifang535 add
    # loss = loss3
    if epoch_id <= 50:  # lifang535: for new attack
        loss = loss3 # lifang535 add # lifang535: for new attack
    else:  # lifang535: for new attack
        loss = loss1+loss4+loss3+loss2  # lifang535: for new attack
    # loss = loss1+loss4+loss3+10*loss2 # lifang535 add
    
    loss.requires_grad_(True)
    adam_opt.zero_grad()
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    count = (scores > 0.25).sum()
    print('loss',loss.item(),'loss_1',loss1.item(),'loss_2',loss2.item(),'loss_3',loss4.item(),'count:',count.item())
    return bx


class SingleAttack:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, image_list, image_name_list, img_size, epochs=None, device=None, results_dict={}): # args, dataloader, img_size, confthre, nmsthre, num_classes):
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
        # self.pipeline = IF_PIPELINE
        self.epochs = epochs
        self.device = device  
        self.results_dict = results_dict    
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        self.model.eval()
        self.names = CONSTANTS.DETR_DICT
        self.learning_rate = 0.001
        
        self.clean_count = -1
        
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
        # model = model.eval()
        # if half:
        #     model = model.half()
        # ids = []
        # data_list = []
        # results = []
        # video_names = defaultdict()
        # progress_bar = tqdm if is_main_process() else iter

        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
        #     model(x)
        #     model = model_trt
            
        frame_id = 0
        total_l1 = 0
        total_l2 = 0
        strategy = 1
        max_tracker_num = int(8)
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(self.device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(self.device)
        # for cur_iter, (imgs,path) in enumerate(
        #     progress_bar(self.dataloader)
        #     ):

        inputs = self.image_processor(images=imgs, return_tensors="pt").to(self.device)
        imgs = inputs["pixel_values"]
        
        frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(self.device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        adam_opt = Adam([bx], lr=self.learning_rate, amsgrad=True)
        
        imgs = util.denormalize(imgs)
        
        imgs = imgs.type(tensor_type)
        imgs = imgs.to(self.device)
        #(1,23625,6)
        max_count = -1.0
        for iter in tqdm(range(self.epochs)):
            epoch_id = iter # lifang535: for new attack
            
            added_imgs = imgs+bx
            
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            added_imgs.clamp_(min=0, max=1)
            # input_imgs = (added_imgs - rgb_means)/std
            if half:
                input_imgs = input_imgs.half()
            # outputs = model(input_imgs) # lifang535 remove
           
            # outputs = model(added_imgs) # lifang535 add
            
            # noise_imgs = add_gaussian_noise(added_imgs) # lifang535: robust_test
            # noise_imgs = add_spatial_smoothing(added_imgs) # lifang535: robust_test
            result = self.model(added_imgs) # lifang535 add
            target_size = [imgs.shape[2:] for _ in range(1)]
            self.img_height, self.img_width = target_size[0][0], target_size[0][1]
            outputs = self.image_processor.post_process_object_detection(result, 
                                                                         threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                                         target_sizes = target_size)[0]

            scores = outputs["scores"]
            labels = outputs["labels"]
            count = (scores >= 0.25).sum() # original: > 0.3
            from overload_attack import generate_mask
            if iter == 0:
                mask = generate_mask(outputs,added_imgs.shape[3],added_imgs.shape[2]).to(self.device)

            bx, count = self.run_attack(outputs, result, bx, strategy, max_tracker_num, adam_opt, epoch_id, count, mask, self.device)
            # bx, count = _run_attack(None, outputs, bx, strategy, max_tracker_num, mask)
            max_count = max(count, max_count)
            if count > max_count:
                max_count = count
                best_img = added_imgs.clone()  # Store the current image
                best_result = result.clone()  # Store the current result
            
            target_size = [imgs.shape[2:] for _ in range(1)]

            if type(count) != type(1) :
                count = count.item()
            if type(max_count) != type(1):
                max_count = max_count.item()
            tqdm.write(f"image_name: {image_name} iter: {iter} count: {count} clean_count: {self.clean_count}")
            tqdm.write(f"label: {labels}")
        
        if args.pipeline_name == "caption":


            start_time = time.perf_counter()
            with torch.no_grad():
                downstream_scores, downstream_labels, downstream_boxes = util.parse_prediction(outputs)
                for box in downstream_boxes:
                    cropped_img = util.crop_img(image_tensor=added_imgs, box=box)
                    # pdb.set_trace()
                    # _result = self.model(added_imgs) 
                    # _target_size = [imgs.shape[2:] for _ in range(1)]
                    # self.image_processor.post_process_object_detection(_result, 
                    #                                                     threshold = CONSTANTS.POST_PROCESS_THRESH, 
                    #                                                     target_sizes = _target_size)[0]

                    ms_captioning = MSCaptioning(device=self.device)
                    ms_captioning.load_processor_checkpoint()
                    ms_captioning.load_model()
                    caption = ms_captioning.inference(added_imgs)
                    # Stop measuring time
                    end_time = time.perf_counter()
                
                # Calculate elapsed time
            elapsed_time = (end_time - start_time) * 1000
        else:
            elapsed_time = -1
            
        self.results_dict[f"image_{image_name}"] = {"clean_bbox_num": int(self.clean_count), "corrupted_bbox_num": int(max_count), "inference time": round(elapsed_time, 2)}

        if strategy == max_tracker_num-1:
            strategy = 0
        else:
            strategy += 1
        print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        
        
        # input_path = f"{input_dir}/{image_name}"
        # output_path = f"{output_dir}/{image_name}"
        # cv2.imwrite(output_path, added_blob) # lifang535: 这个 attack 效果似乎不受小数位损失影响
        
        # print(f"saved image to {output_path}")
        # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms = infer(input_path)
        # _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _target_num_after_nms = infer(output_path)
        
        # # logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, {attack_object}_num_after_nms: {target_num_after_nms}, _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _{attack_object}_num_after_nms: {_target_num_after_nms}")
        # logger.info(f"{objects_num_before_nms} {objects_num_after_nms} {person_num_after_nms} {target_num_after_nms} {_objects_num_before_nms} {_objects_num_after_nms} {_person_num_after_nms} {_target_num_after_nms}")

        
        # save_dir = path[0].replace("ori_img.jpg", "rao_img_3.png")
        # result_dir = os.path.dirname(save_dir)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        #     print(save_dir)
        # cv2.imwrite(save_dir, added_blob)
        # save_dir = path[0].replace("ori_img.jpg", "rao_3.png")
        # bxaaa = torch.clamp(bx*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # print(np.max(bxaaa))
        # print(bxaaa.shape)
        # bxaaa = bxaaa[..., ::-1]
        # cv2.imwrite(save_dir, bxaaa)
        # outputs = outputs.unsqueeze(0)
        # outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        # #scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
        # outputs = outputs[0].detach().cpu().numpy()
        # detections = outputs[:, :7]
        # #detections[:, :4] /= scale
        # aaa_img = torch.clamp(added_imgs*255,0,255)
        # print(len(detections))
        # hua_img = cv2.imread('./tutu/kuang_2.png')
        # for det in detections:
        #     left, top, right, bottom, score = det[:5]
        #     # 将坐标转换为整数
        #     left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        #     # 在图像上绘制矩形
        #     cv2.rectangle(hua_img, (left, top), (right, bottom), (0, 255, 0), 2)
        # result_file_path = path[0].replace("ori_img.jpg", "kuang_3.png")

        # hua_2_img = np.ascontiguousarray(added_blob)
        # cv2.imwrite(result_file_path, hua_img)
        # for det in detections:
        #     left, top, right, bottom, score = det[:5]
        #     # 将坐标转换为整数
        #     left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        #     # 在图像上绘制矩形
        #     cv2.rectangle(hua_2_img, (left, top), (right, bottom), (0, 0, 255), 2)
        # result_file_path = path[0].replace("ori_img.jpg", "kuang_4.png")
        # cv2.imwrite(result_file_path, hua_2_img)
        
        # print(l1_norm.item(),l2_norm.item())              
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        # print(mean_l1.item(),mean_l2.item())
        del bx
        del adam_opt
        del outputs
        del imgs
        del count

        return mean_l1,mean_l2

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
        for index, example in tqdm(enumerate(self.image_list), total=self.image_list.__len__(), desc="Processing COCO data"):
            
            image_id, image, width, height, bbox_id, category, gt_boxes, area = util.parse_example(example)
            assert len(bbox_id) == len(category) == len(gt_boxes) == len(area)
            self.clean_count = len(bbox_id)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            mean_l1, mean_l2 = self.evaluate(image, image_id)
            continue
        
        return
    
        for image, image_name in zip(self.image_list, self.image_name_list):
            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(device).float()
            image /= 255.0

            if len(image.shape) == 3:
                image = image[None]

            # print(f"image.shape = {image.shape}")
            
            mean_l1, mean_l2 = self.evaluate(image, image_name)

    def run_attack(self, output, result, bx, strategy, max_tracker_num, adam_opt, epoch_id, count, mask, device):

        # Apply softmax to compute probabilities
        logits = result.logits[0]
        # probabilities = F.softmax(logits, dim=-1)  
        probabilities = F.sigmoid(logits) 
        max_probs, predicted_classes = probabilities.max(dim=-1)  # Shape: [100]
        
        # outputs = outputs[0][0] # lifang535 add
        
        per_num_b = (25*45)/max_tracker_num
        per_num_m = (50*90)/max_tracker_num
        per_num_s = (100*180)/max_tracker_num

        scores, labels, boxes = util.parse_prediction(output)
        
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[: ,3]- boxes[:, 1]
        
        sel_dets = scores
        sel_height = height.clone()
        sel_width = width.clone()
        sel_aaa = (sel_width/self.img_width) * (sel_height/self.img_height)
        
        loss1_targets = torch.ones_like(sel_dets)
        loss1 = 10*(F.mse_loss(sel_dets, loss1_targets, reduction='sum'))
        
        loss2 = 40*torch.norm(bx, p=2)
        
        loss3_targets = torch.ones_like(scores) # lifang535 remove
        loss3 = 1.0*(F.mse_loss(scores, loss3_targets, reduction='sum'))
        
        loss4 = 100*torch.sum(sel_aaa) # lifang535: 相较于 stra_attack，这里的 loss4 是对所有的 box 的面积求和
        
        target_class_tensor = util.set_target_class(args.target_cls_idx).to(self.device)
        loss5 = 0.1 * F.mse_loss(probabilities, target_class_tensor, reduction='sum')
        
        
        # total_loss =  loss2 + loss3 + loss4 + loss5 + loss1
        CONSTANTS.MAX_COUNT = 100
        good = (count / (self.clean_count+1e-4)) >= 3.0 or count >=60
        G = math.sin(min((epoch_id/100.)*math.pi/2, math.pi/2))
        H = G
        factor_l2 = G
        factor_l3 = 3 - G - H
        factor_l4 = H
        total_loss = factor_l2 * loss2 + factor_l3 * loss3 + factor_l4 * loss4 + loss5
        
        total_loss.requires_grad_(True)
        
        adam_opt.zero_grad()
        total_loss.backward(retain_graph=True)
        bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
        bx.data = -1.5 * bx.grad+ bx.data
        # count = (scores > CONSTANTS.POST_PROCESS_THRESH).sum() 
        # default set as 0.25 = CONSTANTS.POST_PROCESS_THRESH
        tqdm.write(f"factor 2: {factor_l2:.4f} factor 3: {factor_l3:.4f} factor 4: {factor_l4:.4f} ")

        tqdm.write(f"loss: {total_loss:.4f} loss_1: {loss1:.4f} loss_2: {loss2:.4f} loss_3: {loss3:.4f} loss_4: {loss4:.4f} good: {good}")
        tqdm.write("")
        # pdb.set_trace()

        return bx, count   

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        Args:
            patience (int): How many epochs to wait after the last time the loss improved.
            min_delta (float): Minimum change in the loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        """
        Call this method at the end of each epoch to check if training should stop.
        Args:
            loss (float): The current epoch's loss.
        """
        # First epoch or first call
        if self.best_loss is None:
            self.best_loss = loss  # Set the initial best loss
        elif loss > self.best_loss - self.min_delta:
            # If current loss is significantly better (improved by `min_delta`)
            self.best_loss = loss  # Update the best loss
            self.counter = 0       # Reset the patience counter
        else:
            # No significant improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
                self.best_loss = None
                self.counter = 0


# def infer(image_path):
#     image = cv2.imread(image_path)
#     # print(f"image.shape = {image.shape}") # (608, 1088, 3)
#     # print(f"image = {image}")

#     image = image.transpose((2, 0, 1))[::-1]
#     image = np.ascontiguousarray(image)
#     image = torch.from_numpy(image).to(device).float()
#     image /= 255.0
    
#     if len(image.shape) == 3:
#         image = image[None]
    
#     # tensor_type = torch.cuda.FloatTensor
#     # image_tensor = image.type(tensor_type)
#     # image_tensor = image.to(device)
    
#     image_tensor = image
    
#     # print(f"image_tensor = {image_tensor}")
    
#     outputs = model(image_tensor)
    
#     # print(f"outputs = {outputs}")
    
#     outputs = outputs[0].unsqueeze(0)
    
#     # scores = outputs[..., index] * outputs[..., 4]
#     # scores = scores[scores > 0.25]
#     # print(f"len(scores) = {len(scores)}")
#     # objects_num_before_nms = len(scores) # 实际上是 {attack_object} number before NMS
    
#     conf_thres = 0.25 # 0.25  # confidence threshold
#     iou_thres = 0.45  # 0.45  # NMS IOU threshold
#     max_det = 100000    # maximum detections per image
    
#     xc = outputs[..., 4] > 0
#     x = outputs[0][xc[0]]
#     x[:, 5:] *= x[:, 4:5]
#     max_scores = x[:, 5:].max(dim=-1).values
#     objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
#     objects_num_after_nms = 0
#     person_num_after_nms = 0
#     target_num_after_nms = 0
    
#     outputs = non_max_suppression(outputs, conf_thres, iou_thres, max_det=max_det)
        
#     for i, det in enumerate(outputs): # detections per image
#         if len(det):
#             for *xyxy, conf, cls in reversed(det):
#                 c = int(cls)
#                 label = f"{names[c]}"
#                 confidence = float(conf)
#                 confidence_str = f"{confidence}" # f"{confidence:.2f}"
#                 box = [round(float(i), 2) for i in xyxy]
#                 # print(f"Detected {label} with confidence {confidence_str} at location {box}")
#                 if label == "person":
#                     person_num_after_nms += 1
#                 if label == attack_object:
#                     target_num_after_nms += 1
#             objects_num_after_nms = len(det)
#         # print(f"There are {len(det)} objects detected in this image.")
    
#     # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms
#     print(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, {attack_object}_num_after_nms = {target_num_after_nms}")
#     return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, target_num_after_nms


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

import CONSTANTS
import random
from datasets import load_dataset
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import AutoImageProcessor
import pdb
import util
random.seed(42)

if __name__ == "__main__":
    # image_name = "000001.png"
    # input_path = f"original_image/{image_name}"
    # output_path = f"single_attack_image/person_epochs_200/{image_name}"
    # weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    # device = torch.device('cuda:1')
    # model = DetectMultiBackend(weights=weights, device=device)
    # names = model.names
    # infer(output_path)
    # time.sleep(10000000)

    # weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    # weights = "./model/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    # device = torch.device('cuda:1')
    # model = DetectMultiBackend(weights=weights, device=device)
    # names = model.names
    # print(f"names = {names}")
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
    
    device = torch.device('cuda:1')
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()
    names = CONSTANTS.DETR_DICT
    
    attack_object_key = 0 # 0: person, 2: car
    attack_object = names[attack_object_key]
    index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index
    
    # attack_method = f"adaptive_attack" # non_adaptive_attack adaptive_attack animal_test attack_threshold_defense
    # adaptive_attack_for_smoothing adaptive_attack_for_gaussian adaptive_attack_for_threshold
    attack_method = f"adaptive_attack_original" # non_adaptive_attack adaptive_attack animal_test attack_threshold_defense
    
    # for attack_object_key in range(0, 1): # lifang535 add
    for attack_object_key in [0]: # lifang535 add
        # TODO: 一些 label 一旦出现一个，就会出现再出现很多，或许可以先注入，或者增大迭代次数
        # attack_object_key = 0
        attack_object = names[attack_object_key]
        index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

        epochs = 200
        learning_rate = 0.01 #0.07
        
        # logger_path = f"adaptive_attack/log/{attack_object}_epochs_{epochs}.log"
        # logger_dir = f"{attack_method}/log/epochs_{epochs}"
        # if not os.path.exists(logger_dir):
        #     os.makedirs(logger_dir)
        # logger_path = f"{logger_dir}/{attack_object_key}_{attack_object}.log"
        
        # logger = create_logger(f"{attack_method}_{attack_object}_epochs_{epochs}", logger_path, logging.INFO)

        # input_dir = "animal/input"
        # output_dir = "animal/output"
        input_dir = "original_image"
        output_dir = f"single_attack_image/{attack_object}_epochs_{epochs}"
        
        # start_time = time.time()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        img_size = (608, 1088)
        image_list, image_name_list = dir_process(input_dir)
        
        sa = SingleAttack(
            image_list=image_list,
            image_name_list=None,
            img_size=img_size,
            epochs=epochs,
            device=None
        )
        
        sa.run()
        
        # single_attack(image_list, image_name_list)
        
        # TODO: 测一下哪步时延长
