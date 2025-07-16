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

    model_1.eval(); model_2.eval()
    for p in model_1.parameters():
        p.requires_grad = False
    for p in model_2.parameters():
        p.requires_grad = False

    log_dict = {}
    for example in tqdm(coco_val_set):
        image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area = \
            object_detection_preprocess(example, processor_1, device=device)
        
        # ### MODIFICATION START: Initialize perturbation or patch ###
        # Initialize adversarial perturbation `bx`. Can be a full-image perturbation or a smaller patch.
        if args.use_patch:
            if args.patch_size > min(width, height):
                print(f"Warning: Patch size ({args.patch_size}) is larger than image dimensions ({width}x{height}). Skipping image {image_id}.")
                continue
            patch_shape = (img_tensor.shape[0], img_tensor.shape[1], args.patch_size, args.patch_size)
            bx = torch.rand(patch_shape, device=device, requires_grad=True) # Start with a random patch
        else:
            bx = torch.zeros_like(img_tensor).to(device).requires_grad_(True)
        # ### MODIFICATION END ###

        target_size = get_target_size(img_tensor)
        final_loss1, final_loss2 = None, None

        for i in range(args.num_it):
            num_dets_1, num_dets_2 = 0, 0

            # ### MODIFICATION START: Apply perturbation or patch ###
            # Apply current perturbation to the image
            if args.use_patch:
                img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
                patch_h, patch_w = bx.shape[2], bx.shape[3]
                
                # For robustness, place the patch at a random location in each iteration
                start_h = random.randint(0, img_h - patch_h)
                start_w = random.randint(0, img_w - patch_w)

                perturbed_img = img_tensor.clone()
                perturbed_img[:, :, start_h:start_h+patch_h, start_w:start_w+patch_w] += bx
            else:
                # Original full-image perturbation
                perturbed_img = img_tensor + bx
            # ### MODIFICATION END ###

            raw1 = model_1(perturbed_img)
            results1 = processor_1.post_process_object_detection(raw1, target_sizes=target_size, threshold=args.thres_1)
            num_dets_1 = get_num_dets(results1)
            loss_1 = object_detection_loss(raw1, args.target_1)
            loss_2 = 0.0
            
            cropped_tensors = []
            if num_dets_1 > 0:
                for res in results1:
                    crops = crop_interpolate_pad(perturbed_img, res, device, interpolate_size=(320, 320))
                    if len(crops) > 0:
                        cropped_tensors.extend(crops)
            
            if len(cropped_tensors) > 0:
                cat_cropped_tensor = torch.cat(cropped_tensors, dim=0)
                cat_target_size = get_target_size(cat_cropped_tensor)
                raw2 = model_2(cat_cropped_tensor)
                results2 = processor_2.post_process_object_detection(raw2, target_sizes=cat_target_size, threshold=args.thres_2)
                num_dets_2 = get_num_dets(results2)
                loss_2 = object_detection_loss(raw2, args.target_2)

            max_dets_1 = args.num_q_1
            frac = float(num_dets_1) / float(max_dets_1)
            w1, w2 = 1.5 - frac, 0.5 + frac
            total_loss = w1 * loss_1 + w2 * loss_2
            torch.autograd.backward(total_loss)

            # ### MODIFICATION START: Corrected gradient ascent update ###
            # Normalize and apply perturbation update (gradient ascent to maximize loss)
            # This logic works for both the full perturbation and the patch (`bx`)
            bx.grad = bx.grad / (torch.norm(bx.grad, p=2) + 1e-20)
            # Perform gradient ascent to maximize the loss (corrected from original code)
            bx.data = bx.data + (args.step_size * bx.grad)
            # Constrain perturbation magnitude for imperceptibility
            bx.data.clamp_(-args.eps, args.eps)
            bx.grad.zero_()
            # ### MODIFICATION END ###

            # Store final iteration losses for logging
            final_loss1 = loss_1.item() if isinstance(loss_1, torch.Tensor) else float(loss_1)
            final_loss2 = loss_2.item() if isinstance(loss_2, torch.Tensor) else float(loss_2)

        # Log results for this image
        log_dict[f"{image_id}_layer1_num_detections"] = num_dets_1
        log_dict[f"{image_id}_layer2_num_detections"] = num_dets_2
        log_dict[f"{image_id}_L1_loss"] = final_loss1
        log_dict[f"{image_id}_L2_loss"] = final_loss2
        log_dict[f"{image_id}_L_total"] = final_loss1 + final_loss2 if final_loss1 is not None and final_loss2 is not None else None

    output_path = os.path.join(args.output_dir, "final_results.json")
    with open(output_path, "w") as f:
        json.dump(log_dict, f, indent=4)


def worker(gpu, args):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    coco_val_set = ms_coco_init(args.test_size)
    chunks = np.array_split(list(range(len(coco_val_set))), args.ngpus)
    local_dataset = Subset(coco_val_set, [int(idx) for idx in chunks[gpu]])

    model_1, processor_1 = object_detection_init(ckpt=args.ckpt_1, num_q=args.num_q_1, device=device)
    model_2, processor_2 = object_detection_init(ckpt=args.ckpt_2, num_q=args.num_q_2, device=device)
    model_1.eval(); model_2.eval()
    for p in model_1.parameters(): p.requires_grad = False
    for p in model_2.parameters(): p.requires_grad = False

    log_dict = {}
    for example in tqdm(local_dataset, desc=f"GPU {gpu}", position=gpu):
        image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area = \
            object_detection_preprocess(example, processor_1, device=device)
        
        # ### MODIFICATION START: Initialize perturbation or patch ###
        if args.use_patch:
            if args.patch_size > min(width, height):
                continue # Skip images smaller than the patch
            patch_shape = (img_tensor.shape[0], img_tensor.shape[1], args.patch_size, args.patch_size)
            bx = torch.zeros(patch_shape, device=device, requires_grad=True)
        else:
            bx = torch.zeros_like(img_tensor).to(device).requires_grad_(True)
        # ### MODIFICATION END ###

        target_size = get_target_size(img_tensor)
        log_dict[f"id_{image_id}"] = {}

        for i in range(args.num_it):
            # ### MODIFICATION START: Apply perturbation or patch ###
            if args.use_patch:
                if args.patch_placement == "random":
                    img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
                    patch_h, patch_w = bx.shape[2], bx.shape[3]
                    start_h = random.randint(0, img_h - patch_h)
                    start_w = random.randint(0, img_w - patch_w)
                    perturbed_img = img_tensor.clone()
                    perturbed_img[:, :, start_h:start_h+patch_h, start_w:start_w+patch_w] += bx
                elif args.patch_placement == "center":
                    img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
                    patch_h, patch_w = bx.shape[2], bx.shape[3]
                    start_h = (img_h - patch_h) // 2
                    start_w = (img_w - patch_w) // 2
                    perturbed_img = img_tensor.clone()
                    perturbed_img[:, :, start_h:start_h+patch_h, start_w:start_w+patch_w] += bx
            else:
                perturbed_img = img_tensor + bx
            # ### MODIFICATION END ###

            raw1 = model_1(perturbed_img)
            results1 = processor_1.post_process_object_detection(raw1, target_sizes=target_size, threshold=args.thres_1)
            num_dets_1 = get_num_dets(results1)
            loss_1 = object_detection_loss(raw1, args.target_1)
            loss_2 = 0.0

            cropped_tensors = []
            if num_dets_1 > 0:
                for res in results1:
                    crops = crop_interpolate_pad(perturbed_img, res, device, interpolate_size=(320, 320))
                    if len(crops) > 0:
                        cropped_tensors.extend(crops)
            
            if len(cropped_tensors) > 0:
                cat_cropped_tensor = torch.cat(cropped_tensors, dim=0)
                raw2 = model_2(cat_cropped_tensor)
                num_dets_2 = get_num_dets(processor_2.post_process_object_detection(raw2, target_sizes=get_target_size(cat_cropped_tensor), threshold=args.thres_2))
                # Normalize L2 loss by the number of crops to prevent explosion
                loss_2 = object_detection_loss(raw2, args.target_2) / num_dets_1
            else:
                num_dets_2 = 0

            max_dets_1 = args.num_q_1
            frac = float(num_dets_1) / float(max_dets_1)
            w1, w2 = 1.5 - frac, 0.5 + frac
            total_loss = w1 * loss_1 + w2 * loss_2
            torch.autograd.backward(total_loss)

            # ### MODIFICATION START: Corrected gradient ascent update ###
            bx.grad = bx.grad / (torch.norm(bx.grad, p=2) + 1e-20)
            bx.data = bx.data + (args.step_size * bx.grad)
            bx.data.clamp_(-args.eps, args.eps)
            bx.grad.zero_()
            # ### MODIFICATION END ###

            final_loss1 = loss_1.item() if isinstance(loss_1, torch.Tensor) else float(loss_1)
            final_loss2 = loss_2.item() if isinstance(loss_2, torch.Tensor) else float(loss_2)

            log_dict[f"id_{image_id}"][f"iter_{i}"] = {
                "L1_num": num_dets_1, "L2_num": num_dets_2,
                "L1_loss": final_loss1, "L2_loss": final_loss2,
                "L_total": final_loss1 + final_loss2 if final_loss1 is not None and final_loss2 is not None else None
            }

    out_path = os.path.join(args.output_dir, f"RANK_{gpu}.json")
    with open(out_path, "w") as f:
        json.dump(log_dict, f, indent=4)
    print(f"[GPU {gpu}] done, results saved to {out_path}")

if __name__ == "__main__":
    # --- Default settings ---
    TEST_SIZE = None
    PARALLEL = False
    NUM_ITERATIONS = 200
    CKPT_1, CKPT_2 = 0, 0
    NUM_Q_1, NUM_Q_2 = 100, 100
    THRES_1, THRES_2 = 0.5, 0.5
    TARGET_1, TARGET_2 = 2, 2
    OUTPUT_DIR = "./saved/attacked/"
    
    # ### MODIFICATION START: Add new defaults for patch attack ###
    USE_PATCH = False
    PATCH_SIZE = 50
    EPSILON = 0.04
    STEP_SIZE = 1.5
    # ### MODIFICATION END ###

    parser = argparse.ArgumentParser(description="Two-Stage Object Detection Attack (Efficiency-Oriented)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--test_size", type=int, default=TEST_SIZE)
    parser.add_argument("--parallel", action="store_true", default=PARALLEL)
    parser.add_argument("--num_it", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--ckpt_1", type=int, default=CKPT_1)
    parser.add_argument("--ckpt_2", type=int, default=CKPT_2)
    parser.add_argument("--num_q_1", type=int, default=NUM_Q_1)
    parser.add_argument("--num_q_2", type=int, default=NUM_Q_2)
    parser.add_argument("--thres_1", type=float, default=THRES_1)
    parser.add_argument("--thres_2", type=float, default=THRES_2)
    parser.add_argument("--target_1", type=int, default=TARGET_1)
    parser.add_argument("--target_2", type=int, default=TARGET_2)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--patch_placement", type=str, default="random", choices=["random", "center"],
                        help="Placement strategy for the patch: 'random' or 'center'.")

    # ### MODIFICATION START: Add new arguments for patch attack ###
    parser.add_argument("--use_patch", action="store_true", default=USE_PATCH,
                        help="Enable patch-based attack instead of full-image perturbation.")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE,
                        help="Side length of the square adversarial patch.")
    parser.add_argument("--eps", type=float, default=EPSILON,
                        help="Maximum perturbation value (epsilon) for clamping.")
    parser.add_argument("--step_size", type=float, default=STEP_SIZE,
                        help="Step size (learning rate) for gradient ascent.")
    # ### MODIFICATION END ###

    args = parser.parse_args()
    args.output_dir = os.path.join(
        args.output_dir,
        f"Q1_{args.num_q_1}_T1_{args.thres_1}_Q2_{args.num_q_2}_T2_{args.thres_2}",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.parallel:
        ngpus = torch.cuda.device_count()
        args.ngpus = ngpus
        mp.spawn(worker, nprocs=ngpus, args=(args,))
    else:
        main(args)