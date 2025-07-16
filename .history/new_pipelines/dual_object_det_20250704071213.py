import os
import sys
import pdb
import json
import torch
import random
import argparse
import datetime
import deepspeed
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from deepspeed import init_inference

sys.path.append("../")
import utils
import model_zoo
from pipeline_utils import (
    ms_coco_init,
    get_num_dets,
    get_target_size,
    crop_interpolate_pad,
    object_detection_init,
    object_detection_loss,
    object_detection_inference,
    object_detection_preprocess,
)

utils.set_all_seeds(0)

ds_config = {
    "train_batch_size": 1,
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"},
    },
    "fp16": {"enabled": True},
}

# Note: The attack is *efficiency-targeted*, aiming to maximize computational cost.
# We will use an adaptive weighting for the two loss components (L1 for M1, L2 for M2).

def main(args):
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coco_val_set = ms_coco_init(args.test_size)
    model_1, processor_1 = object_detection_init(ckpt=args.ckpt_1, num_q=args.num_q_1, device=device)
    model_2, processor_2 = object_detection_init(ckpt=args.ckpt_2, num_q=args.num_q_2, device=device)

    model_1.eval();  model_2.eval()
    for p in model_1.parameters():
        p.requires_grad = False
    for p in model_2.parameters():
        p.requires_grad = False

    log_dict = {}
    for example in tqdm(coco_val_set):
        image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area = \
            object_detection_preprocess(example, processor_1, device=device)
        # Initialize adversarial perturbation `bx` (same shape as img_tensor)
        bx = torch.zeros_like(img_tensor).to(device).requires_grad_(True)
        target_size = get_target_size(img_tensor)

        # We will record final losses for logging
        final_loss1 = None
        final_loss2 = None

        for i in range(args.num_it):
            # Number of detections in this iteration (for adaptive weighting and info)
            num_dets_1, num_dets_2 = 0, 0

            # Apply current perturbation
            perturbed_img = img_tensor + bx
            # Run first-stage model (M1)
            raw1 = model_1(perturbed_img)
            results1 = processor_1.post_process_object_detection(raw1, target_sizes=target_size, threshold=args.thres_1)
            num_dets_1 = get_num_dets(results1)
            # Compute loss for M1: encourage as many detections as possible (efficiency attack)
            loss_1 = object_detection_loss(raw1, args.target_1)  # In our design, target_1 is chosen to maximize detections.
            # (For example, object_detection_loss might push predictions toward a specific object class 
            # to avoid 'no object' outputs, effectively increasing detection count.)

            loss_2 = 0.0
            # Process each detection from M1 through second-stage model (M2)
            cropped_tensors = []
            for res in results1:
                crops = crop_interpolate_pad(perturbed_img, res, device, interpolate_size=(320, 320))
                if len(crops) == 0:
                    continue
                # We batch all cropped regions for efficiency (if multiple detections)
                cropped_tensors.extend(crops)
            if len(cropped_tensors) > 0:
                cat_cropped_tensor = torch.cat(cropped_tensors, dim=0)
                cat_target_size = get_target_size(cat_cropped_tensor)
                raw2 = model_2(cat_cropped_tensor)
                results2 = processor_2.post_process_object_detection(raw2, target_sizes=cat_target_size, threshold=args.thres_2)
                num_dets_2 = get_num_dets(results2)
                # Compute loss for M2: similarly encourage detections in each cropped region
                loss_2 = object_detection_loss(raw2, args.target_2)
                # If multiple crops were processed in batch, consider extending loss if needed:
                # (Depending on implementation, object_detection_loss might already handle batched input appropriately.)

            # Adaptive weighting: adjust w1 and w2 based on current M1 detections
            max_dets_1 = args.num_q_1  # maximum possible detections from M1 (number of queries)
            frac = float(num_dets_1) / float(max_dets_1)  # fraction of M1 slots that are filled
            # We ensure w1 + w2 ~= 2 for consistency with original scale, and neither is zero
            w1 = 1.5 - frac
            w2 = 0.5 + frac

            # Combined loss with adaptive weights
            total_loss = w1 * loss_1 + w2 * loss_2
            # Backpropagate the combined loss
            torch.autograd.backward(total_loss)

            # Normalize and apply perturbation update (gradient ascent on loss to maximize it)
            bx.grad = bx.grad / (torch.norm(bx.grad, p=2) + 1e-20)
            bx.data = bx.data + (-1.5 * bx.grad)  # step size of 1.5 (scaling the perturbation in gradient direction)
            bx.data.clamp_(-0.04, 0.04)  # constrain perturbation magnitude for imperceptibility
            # Clear gradients for next iteration
            bx.grad.zero_()

            # Optionally, print debug info
            print(f"Iteration {i+1}/{args.num_it} – num_dets_1: {num_dets_1}, num_dets_2: {num_dets_2}, w1: {w1:.2f}, w2: {w2:.2f}")

            # Store final iteration losses for logging
            final_loss1 = loss_1.item() if isinstance(loss_1, torch.Tensor) else float(loss_1)
            final_loss2 = loss_2.item() if isinstance(loss_2, torch.Tensor) else float(loss_2)
            # (We convert to Python float for logging to avoid JSON issues with tensors.)

        # End of attack iterations for this image – log results
        log_dict[f"{image_id}_layer1_num_detections"] = num_dets_1
        log_dict[f"{image_id}_layer2_num_detections"] = num_dets_2
        log_dict[f"{image_id}_L1_loss"] = final_loss1
        log_dict[f"{image_id}_L2_loss"] = final_loss2
        log_dict[f"{image_id}_L_total"] = final_loss1 + final_loss2 if final_loss1 is not None and final_loss2 is not None else None

    # Save the log as JSON
    output_path = os.path.join(args.output_dir,
        f"NUM_Q_1_{args.num_q_1}_THRES_1_{args.thres_1}_NUM_Q_2_{args.num_q_2}_THRES_2_{args.thres_2}.json")
    with open(output_path, "w") as f:
        json.dump(log_dict, f, indent=4)


def worker(gpu, args):
    # This function is similar to main(), but for parallel execution across GPUs.
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    coco_val_set = ms_coco_init(args.test_size)
    total_size = len(coco_val_set)
    all_indices = list(range(total_size))
    chunks = np.array_split(all_indices, args.ngpus)
    my_indices = [int(idx) for idx in chunks[gpu]]
    local_dataset = Subset(coco_val_set, my_indices)

    model_1, processor_1 = object_detection_init(ckpt=args.ckpt_1, num_q=args.num_q_1, device=device)
    model_2, processor_2 = object_detection_init(ckpt=args.ckpt_2, num_q=args.num_q_2, device=device)
    model_1.eval();  model_2.eval()
    for p in model_1.parameters():
        p.requires_grad = False
    for p in model_2.parameters():
        p.requires_grad = False

    log_dict = {}
    for example in tqdm(local_dataset, desc=f"GPU {gpu}", position=gpu):
        image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area = \
            object_detection_preprocess(example, processor_1, device=device)
            
        bx = torch.zeros_like(img_tensor).to(device).requires_grad_(True)
        target_size = get_target_size(img_tensor)

        final_loss1 = None
        final_loss2 = None
        
        log_dict[f"id_{image_id}"] = {}

        for i in range(args.num_it):
            num_dets_1, num_dets_2 = 0, 0
            
            perturbed_img = img_tensor + bx
            
            raw1 = model_1(perturbed_img)
            results1 = processor_1.post_process_object_detection(raw1, target_sizes=target_size, threshold=args.thres_1)
            num_dets_1 = get_num_dets(results1)
            loss_1 = object_detection_loss(raw1, args.target_1)
            loss_2 = 0.0
            
            cropped_tensors = []
            for res in results1:
                crops = crop_interpolate_pad(perturbed_img, res, device, interpolate_size=(320, 320))
                if len(crops) == 0:
                    continue
                cropped_tensors.extend(crops)
            if len(cropped_tensors) > 0:
                cat_cropped_tensor = torch.cat(cropped_tensors, dim=0)
                cat_target_size = get_target_size(cat_cropped_tensor)
                raw2 = model_2(cat_cropped_tensor)
                results2 = processor_2.post_process_object_detection(raw2, target_sizes=cat_target_size, threshold=args.thres_2)
                num_dets_2 = get_num_dets(results2)
                loss_2 = object_detection_loss(raw2, args.target_2) / num_dets_1

            max_dets_1 = args.num_q_1
            frac = float(num_dets_1) / float(max_dets_1)
            w1 = 1.5 - frac
            w2 = 0.5 + frac

            total_loss = w1 * loss_1 + w2 * loss_2
            torch.autograd.backward(total_loss)
            bx.grad = bx.grad / (torch.norm(bx.grad, p=2) + 1e-20)
            bx.data = bx.data + (-1.5 * bx.grad)
            bx.data.clamp_(-0.04, 0.04)
            bx.grad.zero_()

            # if gpu == 0:  # print debug from GPU 0 for example
            #     print(f"[GPU{gpu}] Iter {i+1}/{args.num_it} – num_dets_1: {num_dets_1}, num_dets_2: {num_dets_2}, w1: {w1:.2f}, w2: {w2:.2f}")
            final_loss1 = loss_1.item() if isinstance(loss_1, torch.Tensor) else float(loss_1)
            final_loss2 = loss_2.item() if isinstance(loss_2, torch.Tensor) else float(loss_2)

            log_dict[f"id_{image_id}"][f"iter_{i}"] = {}
            log_dict[f"id_{image_id}"][f"iter_{i}"][f"L1_num"] = num_dets_1
            log_dict[f"id_{image_id}"][f"iter_{i}"][f"L2_num"] = num_dets_2
            log_dict[f"id_{image_id}"][f"iter_{i}"][f"L1_loss"] = final_loss1
            log_dict[f"id_{image_id}"][f"iter_{i}"][f"L2_loss"] = final_loss2
            log_dict[f"id_{image_id}"][f"iter_{i}"][f"L_total"] = final_loss1 + final_loss2 if final_loss1 is not None and final_loss2 is not None else None

    out_path = os.path.join(
        args.output_dir,
        f"RANK_{gpu}.json"
    )
    with open(out_path, "w") as f:
        json.dump(log_dict, f, indent=4)
    print(f"[GPU {gpu}] done, results saved to {out_path}")


if __name__ == "__main__":
    TEST_SIZE = None
    PARALLEL = False
    NUM_ITERATIONS = 200
    CKPT_1 = 0
    CKPT_2 = 0
    NUM_Q_1 = 100
    NUM_Q_2 = 100
    THRES_1 = 0.5
    THRES_2 = 0.5
    TARGET_1 = 2   # we use target class 2 as a representative object class to maximize (not accuracy-driven, just to induce detections)
    TARGET_2 = 2
    OUTPUT_DIR = "./saved/attacked/"

    parser = argparse.ArgumentParser(description="Two-Stage Object Detection Attack (Efficiency-Oriented)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--test_size", type=int, default=TEST_SIZE, help="Number of samples to test on")
    parser.add_argument("--parallel", action="store_true", default=PARALLEL, help="Run in parallel mode")
    parser.add_argument("--num_it", type=int, default=NUM_ITERATIONS, help="Number of attack iterations")
    parser.add_argument("--ckpt_1", type=int, default=CKPT_1, help="Checkpoint for the first model")
    parser.add_argument("--ckpt_2", type=int, default=CKPT_2, help="Checkpoint for the second model")
    parser.add_argument("--num_q_1", type=int, default=NUM_Q_1, help="Number of queries for model 1 (M1)")
    parser.add_argument("--num_q_2", type=int, default=NUM_Q_2, help="Number of queries for model 2 (M2)")
    parser.add_argument("--thres_1", type=float, default=THRES_1, help="Confidence threshold for M1 detections")
    parser.add_argument("--thres_2", type=float, default=THRES_2, help="Confidence threshold for M2 detections")
    parser.add_argument("--target_1", type=int, default=TARGET_1, help="Target class (or objective) for the first model")
    parser.add_argument("--target_2", type=int, default=TARGET_2, help="Target class (or objective) for the second model")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the results")

    args = parser.parse_args()
    args.output_dir = os.path.join(
        args.output_dir,
        f"Q1_{args.num_q_1}_T1_{args.thres_1}"
        f"_Q2_{args.num_q_2}_T2_{args.thres_2}",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.parallel:
        ngpus = torch.cuda.device_count()
        args.ngpus = ngpus
        mp.spawn(worker, nprocs=ngpus, args=(args,))
    else:
        main(args)