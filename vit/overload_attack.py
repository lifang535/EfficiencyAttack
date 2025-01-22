from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

# from yolox.utils import (
#     gather,
#     is_main_process,
#     postprocess,
#     synchronize,
#     time_synchronized,
#     xyxy2xywh
# )

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

import torchvision

from pathlib import Path

import sys
YOLOV5_FILE = Path(f"../../yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add ROOT to PATH
    
# from models.common import DetectMultiBackend
# from utils.general import Profile, non_max_suppression
from torchvision.utils import save_image

from PIL import Image
import logging

sys.path.append("../captioning")
from ms_captioning import MSCaptioning
from detr import args
import multiprocessing as mp
from multiprocessing import Process, Queue
from torchvision.transforms import ToPILImage
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue
from tqdm import tqdm
from torchvision.transforms import ToPILImage
import gc

            
def create_logger(module, filename, level):
    # os.makedirs(logger_path, exist_ok=True)

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

def generate_mask(outputs, x_shape, y_shape): # 第一次推理后生成 mask
    mask_x = 4
    mask_y = 2
    mask = torch.ones(y_shape,x_shape)
    
    conf_thres = 0.0 # confidence threshold # lifang535
    iou_thres = 0.0  # NMS IOU threshold
    max_det = 1000   # maximum detections per image
    # outputs = non_max_suppression(prediction=outputs, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
    
    outputs = outputs["boxes"]
    x_len = int(x_shape / mask_x)
    y_len = int(y_shape / mask_y)
    if outputs is not None:
        for i in range(len(outputs)):
            detection = outputs[i]
            center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2
            # 根据检测框的中心点位置，判断它在哪个区域
            region_x = int(center_x / x_len)
            region_y = int(center_y / y_len)
            
            mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05
    
    # print(f"mask.shape = {mask.shape}")
    # print(f"mask = {mask}")
    
    return mask

# def generate_mask(detection_results, x_shape, y_shape):

#     mask_x = 4
#     mask_y = 2
#     # mask = torch.ones(mask_y,mask_x)  # 初始mask为3*3
#     mask = torch.ones(y_shape,x_shape)
#     print(detection_results.shape)
#     detection_results = detection_results.unsqueeze(0)
#     outputs = postprocess(detection_results, num_classes=1, conf_thre=0.1, nms_thre=0.4)[0]
#     # pred = non_max_suppression(
#     #                 detection_results[0], conf_thres, iou_thres, classes, agnostic_nms)
#     x_len = int(x_shape / mask_x)
#     y_len = int(y_shape / mask_y)
#     if outputs is not None:
#         for i in range(len(outputs)):
#             detection = outputs[i]
#             center_x, center_y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2
#             # 根据检测框的中心点位置，判断它在哪个区域
#             region_x = int(center_x / x_len)
#             region_y = int(center_y / y_len)
            
#             mask[region_y*y_len:(region_y+1)*y_len, region_x*x_len:(region_x+1)*y_len] -= 0.05
    
    
#     return mask

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

def run_attack(outputs,bx, strategy, max_tracker_num, mask):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    # outputs = outputs[0][0]
    scores = outputs["scores"]
    # pdb.set_trace()
    loss2 = 40*torch.norm(bx, p=2)
    # targets = torch.ones_like(scores)
    targets = torch.zeros_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    # loss = loss3#+loss2 # lifang535 remove
    loss = loss3+2*(10000-loss2) # lifang535 add: 可以实现 loss2 的效果; 在相同 loss2 的情况下，overload_attack 似乎比 phantom_attack 效果差（好吧不一定，似乎可以调整）
    # TODO:
    # RQ 1.2: Why small threshold is impractical to mitigate efficiency degradation? (Sec. 3.3)
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -3.5 * mask * bx.grad+ bx.data
    count = (scores >= 0.25).sum() # original: > 0.3
    if __name__ == "__main__":
        tqdm.write(f"loss: {loss.item():.4f}, loss_2: {loss2.item():.4f}, loss_3: {loss3.item():.4f} \ncount: {count.item()}\n")
    return bx

def _run_attack(outputs,result,bx, strategy, max_tracker_num, mask):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    # scores = outputs[:,5] * outputs[:,4] # remove
    
    scores = result["scores"] # add
    
    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores)
    loss3 = F.mse_loss(scores, targets, reduction='sum')
    #TODO:
    #dynamic loss 3
    # loss = loss3 * 100 + loss2
    loss = loss3+2*(10000-loss2)
    # loss = loss3
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    # bx.data = -3.5 * mask * bx.grad+ bx.data
    bx.data = torch.clamp(-3.5 * mask * bx.grad+ bx.data, min=-0.2, max=0.2)
    # pdb.set_trace()

    if __name__ == "__main__":
      pass
    tqdm.write(f"loss: {loss.item():.4f}, loss_2: {loss2.item():.4f}, loss_3: {loss3.item():.4f}")
    return bx

class OverloadAttack:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, model, image_processor, image_list, image_name_list, img_size, results_dict = {}, epochs=-1, device=None):
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
        self.confthre = None
        self.nmsthre = None
        self.num_classes = None
        self.args = None

        self.epochs = epochs
        self.device = device  
        self.results_dict = results_dict    
        self.image_processor = image_processor
        self.model = model.to(self.device)
        self.model.eval()
        self.names = CONSTANTS.DETR_DICT

    def evaluate(
        self,
        imgs,
        image_name,
        clean_count,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        # global model, names
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
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
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
        strategy = 0
        max_tracker_num = int(15)
        
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(self.device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(self.device)

        # image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        inputs = self.image_processor(images=imgs, return_tensors="pt").to(self.device)
        imgs = inputs["pixel_values"]
        # save_image(imgs, "./1.png")
        # a = util.denormalize(imgs)
        # pdb.set_trace()
        # if len(imgs.shape) == 3:
        #     imgs = imgs[None]

        # frame_id += 1
        bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
        bx = bx.astype(np.float32)
        bx = torch.from_numpy(bx).to(self.device).unsqueeze(0)
        bx = bx.data.requires_grad_(True)
        # imgs = imgs.type(tensor_type)
        # imgs = imgs.to(device)
        #(1,23625,6)
        
        # model = model.eval()
        # if half:
        #     model = model.half()
        imgs = util.denormalize(imgs)
        max_count = -1.0

        for iter in tqdm(range(self.epochs)):
            added_imgs = imgs+bx
            l2_norm = torch.sqrt(torch.mean(bx ** 2))
            l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
            added_imgs.clamp_(min=0, max=1)
            # input_imgs = (added_imgs - rgb_means)/std 
            # pdb.set_trace()
            # if half:
            #     input_imgs = input_imgs.half()
            # outputs = model(input_imgs)[0] # lifang535 remove
            # outputs = model(input_imgs) # lifang535 add
            # result = model(input_imgs)
            result = self.model(added_imgs) 
            
            target_size = [imgs.shape[2:] for _ in range(1)]
            outputs = self.image_processor.post_process_object_detection(result, 
                                                                    threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                                    target_sizes = target_size)[0]
            
            _, labels, _ = util.parse_prediction(outputs)
            
            if iter == 0:
                mask = generate_mask(outputs,added_imgs.shape[3],added_imgs.shape[2]).to(self.device)


            # bx = run_attack(outputs, bx, strategy, max_tracker_num, mask)
            bx = _run_attack(None, outputs, bx, strategy, max_tracker_num, mask)
            count = len(labels)
            tqdm.write(f"count: {count}")
            if count > max_count:
                max_count = max(count, max_count)
                # max_count = count
                best_img = added_imgs.clone()  # Store the current image
                best_outputs = {key: value.clone() for key, value in outputs.items()} # Store the current result

        
        if args.pipeline_name == "caption":

            start_time = time.perf_counter()
            with torch.no_grad():
                downstream_scores, downstream_labels, downstream_boxes = util.parse_prediction(best_outputs)
                cropped_list = []
                # Crop images based on bounding boxes
                for i, box in enumerate(downstream_boxes):
                    if downstream_labels[i] == args.target_cls_idx:  # Only consider boxes with label == target
                        cropped_img = util.crop_img(image_tensor=best_img, box=box.int())
                        cropped_list.append((cropped_img, i))  # Store cropped image with its index

                # Perform ms_captioning only for label == 1
                for cropped_img, idx in tqdm(cropped_list, desc="Processing cropped images"):
                    ms_captioning = MSCaptioning(device=self.device)
                    ms_captioning.load_processor_checkpoint()
                    ms_captioning.load_model()
                    caption = ms_captioning.inference(cropped_img)
                    # pdb.set_trace()
                end_time = time.perf_counter()
                
            elapsed_time = (end_time - start_time) * 1000
        else:
            elapsed_time = -1
        if args.pipeline_name == None:
            start_time = time.perf_counter()
            
            best_results = self.model(best_img) 
            _ = self.image_processor.post_process_object_detection(best_results, 
                                                                   threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                                   target_sizes = target_size)[0]
            end_time = time.perf_counter()
            _, best_labels, _ = util.parse_prediction(best_outputs)
            elapsed_time = (end_time - start_time) * 1000
            max_count = best_labels.tolist()
            torch.cuda.empty_cache()
        self.results_dict[f"image_{image_name}"] = {"clean_bbox_num": int(self.clean_count), "corrupted_bbox_num": max_count, "inference time": round(elapsed_time, 2)}

        # pdb.set_trace()
        
        if strategy == max_tracker_num-1:
            strategy = 0
        else:
            strategy += 1
        # print(added_imgs.shape)
        added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        added_blob = added_blob[..., ::-1]
        # added_blob_2 = added_blob_2[..., ::-1]
        
        # save_dir = path[0].replace("dataset", "botsort_overload")
        # result_dir = os.path.dirname(save_dir)
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        #     print(save_dir)
        # cv2.imwrite(save_dir, added_blob)
        
        """
        # infer_tensor(input_imgs) # 10000+
        # infer_tensor(added_imgs) # 2000+
        # input_imgs = torch.clamp(input_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # print("input_imgs.shape = ", input_imgs.shape)
        # print("input_imgs = ", input_imgs)
        # infer_image(input_imgs) # lifang535: 似乎这里的 added_blob 已经错了
        # # 转 int 后再转回来，可能会有问题（问题不在这里）
        # input_imgs = input_imgs.astype(np.uint8)
        # input_imgs = input_imgs.astype(np.float32)
        # print("input_imgs.shape = ", input_imgs.shape)
        # print("input_imgs = ", input_imgs)
        # infer_image(input_imgs)
        
        # print("added_blob.shape = ", added_blob.shape)
        # print("added_blob = ", added_blob)
        # infer_image(added_blob) # lifang535: 似乎这里的 added_blob 已经错了
        # # 转 int 后再转回来，可能会有问题
        # added_blob = added_blob.astype(np.uint8)
        # added_blob = added_blob.astype(np.float32)
        # print("added_blob.shape = ", added_blob.shape)
        # print("added_blob = ", added_blob)
        # infer_image(added_blob)
        """
        
        # input_path = f"{input_dir}/{image_name}"
        # output_path = f"{output_dir}/{image_name}"
        # cv2.imwrite(output_path, added_blob)
        
        # time.sleep(10000000)
        
        
        # print(l1_norm.item(),l2_norm.item())
        total_l1 += l1_norm
        total_l2 += l2_norm
        mean_l1 = total_l1/frame_id
        mean_l2 = total_l2/frame_id
        # print(mean_l1.item(),mean_l2.item())
        
        # print(f"saved image to {output_path}")
        # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms = infer(input_path)
        # _objects_num_before_nms, _objects_num_after_nms, _person_num_after_nms, _car_num_after_nms = infer(output_path)
        
        # logger.info(f"objects_num_before_nms: {objects_num_before_nms}, objects_num_after_nms: {objects_num_after_nms}, person_num_after_nms: {person_num_after_nms}, car_num_after_nms: {car_num_after_nms} -> _objects_num_before_nms: {_objects_num_before_nms}, _objects_num_after_nms: {_objects_num_after_nms}, _person_num_after_nms: {_person_num_after_nms}, _car_num_after_nms: {_car_num_after_nms}")
        # infer(output_path_tiff)
        
        del bx
        del outputs
        del imgs
        
        del added_imgs
        del mask
        # del input_imgs
        del added_blob
        del l1_norm
        del l2_norm

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
        # global model, device, names
        device_id = 0
        for index, example in tqdm(enumerate(self.image_list), total=self.image_list.__len__(), desc="Processing COCO data"):
            
            image_id, image, width, height, bbox_id, category, gt_boxes, area = util.parse_example(example)
            assert len(bbox_id) == len(category) == len(gt_boxes) == len(area)
            self.clean_count = len(bbox_id)

            if image.mode != "RGB":
                image = image.convert("RGB")
            
            
            mean_l1, mean_l2 = self.evaluate(image, image_id, len(bbox_id))
            
            continue
            image = image.transpose((2, 0, 1))[::-1]
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(self.device).float()
            image /= 255.0

            if len(image.shape) == 3:
                image = image[None]

            # print(f"image.shape = {image.shape}")
            
            mean_l1, mean_l2 = self.evaluate(image, image_id)
            
            # lifang535: 更换 device
            device_id += 1
            del model
            del device
            del names
            torch.cuda.empty_cache()
            weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
            device = torch.device(f'cuda:{device_id % 4}')
            model = DetectMultiBackend(weights=weights, device=device)
            names = model.names
            
            


# def infer_tensor(image_tensor): # 0 ~ 1
#     image_tensor = image_tensor.to(device)
#     if len(image_tensor.shape) == 3:
#         image_tensor = image_tensor[None]
        
#     outputs = model(image_tensor)
#     outputs = outputs[0].unsqueeze(0)
    
#     conf_thres = 0.25 # 0.25  # confidence threshold
#     iou_thres = 0.45  # 0.45  # NMS IOU threshold
#     max_det = 1000    # maximum detections per image
    
#     xc = outputs[..., 4] > 0
#     x = outputs[0][xc[0]]
#     x[:, 5:] *= x[:, 4:5]
#     max_scores = x[:, 5:].max(dim=-1).values
#     objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
#     objects_num_after_nms = 0
#     person_num_after_nms = 0
#     car_num_after_nms = 0
    
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
#                 elif label == "car":
#                     car_num_after_nms += 1
#             objects_num_after_nms = len(det)
#         # print(f"There are {len(det)} objects detected in this image.")
    
#     # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
#     tqdm.write(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, car_num_after_nms = {car_num_after_nms}")
#     return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
    
        

# def infer_image(image_array): # BGR, 0 ~ 255
#     image_array = image_array.transpose((2, 0, 1))[::-1]
#     image_array = np.ascontiguousarray(image_array)
#     image_array = torch.from_numpy(image_array).to(device).float()
#     image_array /= 255.0
    
#     if len(image_array.shape) == 3:
#         image_array = image_array[None]
        
#     outputs = model(image_array)
#     outputs = outputs[0].unsqueeze(0)
    
#     conf_thres = 0.25 # 0.25  # confidence threshold
#     iou_thres = 0.45  # 0.45  # NMS IOU threshold
#     max_det = 1000    # maximum detections per image
    
#     xc = outputs[..., 4] > 0
#     x = outputs[0][xc[0]]
#     x[:, 5:] *= x[:, 4:5]
#     max_scores = x[:, 5:].max(dim=-1).values
#     objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
#     objects_num_after_nms = 0
#     person_num_after_nms = 0
#     car_num_after_nms = 0
    
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
#                 elif label == "car":
#                     car_num_after_nms += 1
#             objects_num_after_nms = len(det)
#         # print(f"There are {len(det)} objects detected in this image.")
    
#     # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
#     tqdm.write(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, car_num_after_nms = {car_num_after_nms}")
#     return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms

    

# def infer(image_path):
#     image = cv2.imread(image_path)

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
#     max_det = 1000    # maximum detections per image
    
#     xc = outputs[..., 4] > 0
#     x = outputs[0][xc[0]]
#     x[:, 5:] *= x[:, 4:5]
#     max_scores = x[:, 5:].max(dim=-1).values
#     objects_num_before_nms = len(max_scores[max_scores > 0.25]) # 这个是对的，用最大的 class confidence 筛选
    
#     objects_num_after_nms = 0
#     person_num_after_nms = 0
#     car_num_after_nms = 0
    
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
#                 elif label == "car":
#                     car_num_after_nms += 1
#             objects_num_after_nms = len(det)
#         # print(f"There are {len(det)} objects detected in this image.")
    
#     # objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms
#     tqdm.write(f"objects_num_before_nms = {objects_num_before_nms}, objects_num_after_nms = {objects_num_after_nms}, person_num_after_nms = {person_num_after_nms}, car_num_after_nms = {car_num_after_nms}")
#     return objects_num_before_nms, objects_num_after_nms, person_num_after_nms, car_num_after_nms


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

def _dir_process(img_list_len=None):
    if not img_list_len:
        img_list_len = CONSTANTS.VAL_SUBSET_SIZE
    print(f"using img list length: {img_list_len}")    
    coco_data = load_dataset("detection-datasets/coco", split="val")
    random.seed(42)
    random_indices = random.sample(range(len(coco_data)), img_list_len)
    coco_data = coco_data.select(random_indices)
    
    return coco_data

import CONSTANTS
import random
from datasets import load_dataset
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import AutoImageProcessor
import pdb
import util
# import old_overload
random.seed(42)
if __name__ == "__main__":
    # image_name = "000001.png"
    # input_path = f"original_image/{image_name}"
    # output_path = f"overload_attack_image/person_epochs_200/{image_name}"
    # weights = "../model/yolov5/yolov5n.pt" # yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
    # device = torch.device('cuda:1')
    # model = DetectMultiBackend(weights=weights, device=device)
    # names = model.names
    # infer(output_path)
    # time.sleep(10000000)

    # weights = "../model/yolov5/yolov5n.pt"
    # device = torch.device('cuda:0')
    # model = DetectMultiBackend(weights=weights, device=device)
    # names = model.names
    # print(f"names = {names}")
    
    device = torch.device('cuda:1')
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()
    names = CONSTANTS.DETR_DICT
    # pdb.set_trace()
    
    attack_object_key = 1 # 1: person, 3: car, 22: elephant, 25: giraffe
    attack_object = names[attack_object_key]
    
    index = 5 + attack_object_key # yolov5 输出的结果中，class confidence 对应的 index

    epochs = 200
    
    logger_path = f"log/overload_attack/overload_attack_{attack_object}_epochs_{epochs}.log"
    logger = create_logger(f"overload_attack_{attack_object}_epochs_{epochs}", logger_path, logging.INFO)        
    output_dir = f"overload_attack_image/{attack_object}_epochs_{epochs}"
    
    # start_time = time.time()
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    img_size = (608, 1088)
    image_list = _dir_process(50)
    
    oa = OverloadAttack(
        image_list=image_list,
        image_name_list=None,
        img_size=None,
        epochs=epochs
    )
    
    oa.run()
    
    
    # overload_attack(image_list, image_name_list)
    
    # TODO: 测一下哪步时延长
