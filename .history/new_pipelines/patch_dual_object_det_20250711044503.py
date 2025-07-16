import os
import sys
import pdb
import json
import torch
import argparse
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
sys.path.append("../")
import utils
from pipeline_utils import (
    ms_coco_init,
    get_num_dets,
    get_target_size,
    calculate_cls_loss,
    calculate_iou_loss,
    crop_interpolate_pad,
    object_detection_init,
    calculate_distance_loss,
    object_detection_preprocess,
)


def run_attack(dataset, model_1, processor_1, model_2, processor_2, args, device):
    model_1.eval(); model_2.eval()
    for p in model_1.parameters(): p.requires_grad = False
    for p in model_2.parameters(): p.requires_grad = False

    log_dict = {}
    for example in tqdm(dataset):
        image_id, img_tensor, width, height, bbox_id, category, gt_boxes, area = \
            object_detection_preprocess(example, processor_1, device=device)
        bx = torch.zeros_like(img_tensor).to(device).requires_grad_(True)
        target_size = get_target_size(img_tensor)
        log_dict[f"id_{image_id}"] = {}

        for i in range(args.num_it):
            # prepare mask                
            if args.use_patch:
                if args.patch_placement == "center":
                    h, w = img_tensor.shape[2:]
                    ph = pw = args.patch_size
                    sh, sw = (h - ph) // 2, (w - pw) // 2
                    mask = torch.zeros_like(img_tensor)
                    mask[:, :, sh:sh+ph, sw:sw+pw] = 1.0
                elif args.patch_placement == "random":
                    raise ValueError("Random patch placement is not supported in this attack.")
            else:
                mask = torch.ones_like(img_tensor)

            patch_bbox = torch.tensor([sw / img_tensor.shape[3], sh / img_tensor.shape[2], (sw+pw) / img_tensor.shape[3], (sh+ph) / img_tensor.shape[2]], device=device, dtype=torch.float32)

            perturbed = img_tensor + bx * mask
            # stage 1
            raw1 = model_1(perturbed, output_hidden_states=True)
            decoder_feat = raw1.decoder_hidden_states[-1]
            bbox_pred = model_1.bbox_embed[-1](decoder_feat) 
            res1 = processor_1.post_process_object_detection(raw1, target_sizes=target_size, threshold=args.thres_1)

            cls_loss1 = calculate_cls_loss(raw1, args.target_1)
            distance_loss1 = calculate_distance_loss(raw1, patch_bbox)
            iou_loss1 = calculate_iou_loss(raw1, patch_bbox)

            num1 = get_num_dets(res1)

            with torch.no_grad():
                # crop for stage 2
                crops = []
                for r in res1:
                    c = crop_interpolate_pad(perturbed, r, device,
                                            interpolate_size=(320, 320))
                    if len(c) > 0: crops.extend(c)
                num2 = 0
                if crops:
                    cat = torch.cat(crops, dim=0)
                    ts2 = get_target_size(cat)
                    raw2 = model_2(cat)
                    res2 = processor_2.post_process_object_detection(raw2,
                                target_sizes=ts2, threshold=args.thres_2)
                    num2 = get_num_dets(res2)
                    loss2 = calculate_cls_loss(raw2, args.target_2)

            # adaptive weights
            frac = float(num1) / float(args.num_q_1)
            w1 = 1.5 - frac
            w2 = 0.5 + frac
            # total = w1 * loss1 + w2 * loss2
    
            total = w1 * cls_loss1 + w2 * distance_loss1
            total = distance_loss1
            torch.autograd.backward(total)

            # update perturbation
            grad = bx.grad
            grad = grad / (torch.norm(grad, p=2) + 1e-20)
            bx.data = bx.data - args.step_size * grad
            bx.data.clamp_(-args.eps, args.eps)
            bx.grad.zero_()
            
            
            
            log_dict[f"id_{image_id}"][f"iter_{i}"] = {
                "L1_num": num1, "L2_num": num2,
                "cls_loss1": cls_loss1.item(), "distance_loss1": distance_loss1.item(), "iou_loss1": iou_loss1.item(), "L2_loss": loss2.item(), 
                "L_total": total.item()
            }
    return log_dict


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.set_all_seeds(0)

    coco = ms_coco_init(args.test_size)
    model_1, proc_1 = object_detection_init(ckpt=args.ckpt_1, num_q=args.num_q_1, device=device)
    model_2, proc_2 = object_detection_init(ckpt=args.ckpt_2, num_q=args.num_q_2, device=device)

    logs = run_attack(coco, model_1, proc_1, model_2, proc_2, args, device)

    out = os.path.join(args.output_dir, "final_results.json")
    with open(out, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"Finished main, results saved to {out}")


def worker(gpu, args):
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    utils.set_all_seeds(0)

    full = ms_coco_init(args.test_size)
    idx = list(range(len(full)))
    chunks = np.array_split(idx, args.ngpus)
    subset = Subset(full, [int(i) for i in chunks[gpu]])

    model_1, proc_1 = object_detection_init(ckpt=args.ckpt_1, num_q=args.num_q_1, device=device)
    model_2, proc_2 = object_detection_init(ckpt=args.ckpt_2, num_q=args.num_q_2, device=device)

    logs = run_attack(subset, model_1, proc_1, model_2, proc_2, args, device)

    out = os.path.join(args.output_dir, f"RANK_{gpu}.json")
    with open(out, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"[GPU {gpu}] done, results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Stage Object Detection Attack (Efficiency-Oriented)"
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_it", type=int, default=200)
    parser.add_argument("--ckpt_1", type=int, default=0)
    parser.add_argument("--ckpt_2", type=int, default=0)
    parser.add_argument("--num_q_1", type=int, default=100)
    parser.add_argument("--num_q_2", type=int, default=100)
    parser.add_argument("--thres_1", type=float, default=0.5)
    parser.add_argument("--thres_2", type=float, default=0.5)
    parser.add_argument("--target_1", type=int, default=None)
    parser.add_argument("--target_2", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./saved/attacked/"
    )
    parser.add_argument("--patch_placement", type=str,
                        choices=["random", "center"], default="random")
    parser.add_argument("--use_patch", action="store_true")
    parser.add_argument("--patch_size", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.04)
    parser.add_argument("--step_size", type=float, default=1.5)

    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.parallel:
        args.ngpus = torch.cuda.device_count()
        mp.spawn(worker, nprocs=args.ngpus, args=(args,))
    else:
        main(args)
        
        
        