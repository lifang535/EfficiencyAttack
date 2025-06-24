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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset

sys.path.append("../")
import utils
import model_zoo
from pipeline_utils import *
utils.set_all_seeds(0)


def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    coco_val_set = ms_coco_init(args.test_size)
    model_1, processor_1 = object_detection_init(ckpt=args.ckpt_1, num_q=args.num_q_1, device=device)
    model_2, processor_2 = object_detection_init(ckpt=args.ckpt_2, num_q=args.num_q_2, device=device)
    
    log_dict = {}
    for example in tqdm(coco_val_set):
        
        image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area = object_detection_preprocess(example, processor_1, device=device)
        bx = torch.zeros_like(img_tensor).to(img_tensor).requires_grad_(True)
        target_size = get_target_size(img_tensor)
        
        for i in tqdm(range(args.num_it)):
            num_dets_1, num_dets_2 = 0, 0
            
            bx_img_tensor = img_tensor + bx
                        
            raw = model_1(bx_img_tensor)
            results = processor_1.post_process_object_detection(raw, target_sizes=target_size, threshold=args.thres_1)
            num_dets_1 = get_num_dets(results)
            
            loss_1 = object_detection_loss(raw, args.target_1)
            loss_2 = 0.0
            
            for _, res in enumerate(results):
                cropped_list = crop_interpolate_pad(bx_img_tensor, res, device)
                if len(cropped_list) == 0:
                    continue
                
                cat_cropped_tensor = torch.cat(cropped_list, dim=0)
                cat_target_size = get_target_size(cat_cropped_tensor)
                # cat_results = object_detection_inference(model_2, processor_2, cat_cropped_tensor, cat_target_size, args.thres_2, device=device)
                cat_raw = model_2(cat_cropped_tensor)
                
                cat_results = processor_2.post_process_object_detection(cat_raw, target_sizes=cat_target_size, threshold=args.thres_2)
                num_dets_2 += get_num_dets(cat_results)
                
                subtotal_loss_2 = object_detection_loss(cat_raw, args.target_2)
                loss_2 += subtotal_loss_2 
                
                bx.grad_zero_()
                loss = loss_1 + loss_2
                loss.backward()
                print(loss)
                
                bx.grad = bx.grad / (torch.norm(bx.grad, p=2) + 1e-20)
                bx.data = -1.5 * bx.grad + bx.data
                bx.data.clamp_(-0.04, 0.04)
                torch.cuda.empty_cache() 

            
        pdb.set_trace()
            # for j, cropped_img_tensors in enumerate(cropped_list):
            #     _target_size = get_target_size(cropped_img_tensors)
            #     _results = object_detection_inference(model_2, processor_2, cropped_img_tensors, _target_size, THRES_2, device=device)
            #     _num_dets = get_num_dets(_results)
            #     num_dets_2 += _num_dets
        
        log_dict[f"{image_id}_layer_1_num_detections"] = num_dets_1
        log_dict[f"{image_id}_layer_2_num_detections"] = num_dets_2
        
        
    with open(f"./NUM_Q_1_{args.num_q_1}_THRES_1_{args.thres_1}_NUM_Q_2_{args.num_q_2}_THRES_2_{args.thres_2}.json", 'w') as f:
        json.dump(log_dict, f, indent=4)
        

if __name__ == "__main__":
    TEST_SIZE = None
    PARALLEL = False
    NUM_ITERATIONS = 200
    CKPT_1 = 0
    CKPT_2 = 0
    NUM_Q_1 = 300
    NUM_Q_2 = 300
    THRES_1 = 0.5
    THRES_2 = 0.5
    TARGET_1 = 2
    TARGET_2 = 2
    OUTPUT_DIR = "./saved/clean/detr"
    
    parser = argparse.ArgumentParser(description="Object Detection Clean Run")
    parser.add_argument("--test_size", type=int, default=TEST_SIZE, help="Number of samples to test on")
    parser.add_argument("--parallel", action='store_true', default=PARALLEL, help="Run in parallel mode")
    parser.add_argument("--num_it", type=int, default=NUM_ITERATIONS, help="Number of iterations for the attack")
    parser.add_argument("--ckpt_1", type=int, default=CKPT_1, help="Checkpoint for the first model")
    parser.add_argument("--ckpt_2", type=int, default=CKPT_2, help="Checkpoint for the second model")
    parser.add_argument("--num_q_1", type=int, default=NUM_Q_1, help="Number of queries for the model")
    parser.add_argument("--num_q_2", type=int, default=NUM_Q_2, help="Number of queries for the model")
    parser.add_argument("--thres_1", type=float, default=THRES_1, help="Threshold for object detection")
    parser.add_argument("--thres_2", type=float, default=THRES_2, help="Threshold for object detection")
    parser.add_argument("--target_1", type=int, default=TARGET_1, help="Target for the first model")
    parser.add_argument("--target_2", type=int, default=TARGET_2, help="Target for the second model")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the results")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
    
    