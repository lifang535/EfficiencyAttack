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
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
import torch.multiprocessing as mp
from torch.utils.data import Subset

sys.path.append("../")
import utils
import model_zoo
utils.set_all_seeds(0)

def ms_coco_init(num_samples=1000):
    coco_data = load_dataset("detection-datasets/coco", split="val")
    random_indices = random.sample(range(len(coco_data)), num_samples)  
    coco_data = coco_data.select(random_indices)
    return coco_data
    
    
def object_detection_init(ckpt=0, num_q=200, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = model_zoo.load_from_pretrained(
        ckpt=ckpt,
        num_q=num_q,
        device=device,
    )
    return model, processor


def object_detection_preprocess(example, processor, device=None):
    image_id, img, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(example)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_tensor = processor(img, return_tensors="pt")["pixel_values"].to(device)
    img_tensor = utils.denormalize(img_tensor)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0).to(device)
    return image_id, img_tensor, (None, )


def get_target_size(img_tensor):
    _, _, h, w = img_tensor.shape
    minimum_size = 128

    length = max(h, w)
    div = length // minimum_size
    padded_length = min((div + 1) * minimum_size, 640)
    target_size =  [torch.Size([padded_length, padded_length])]
    
    return target_size
    
    
def object_detection_inference(model, processor, img_tensor, target_size, thres):
    _, _, h, w = img_tensor.shape
    if max(h, w) < 128:
        return None
        
    try:
        raw_outputs = model(img_tensor, target_sizes=target_size)
    except RuntimeError as e:
        print(f"[DEBUG] shape={img_tensor.shape}, target_size={target_size}")
        raise e
    
    outputs = processor.post_process_object_detection(
        raw_outputs,
        target_sizes=target_size,
        threshold=thres,
    )[0]

    scores = outputs["scores"]
    # labels = outputs["labels"]
    boxes = outputs["boxes"]
    logits = raw_outputs.logits[0]
    probs = F.sigmoid(logits)
    # combined = torch.cat((boxes, scores.unsqueeze(1), probs[:len(scores)]), dim=1)
    prediction_data = {
        "scores": scores,
        # "labels": labels,
        "boxes": boxes,
        # "logits": logits,
        "probs": probs,
        # "combined": combined,
    }
    
    return prediction_data


def cropper(img_tensor, prediction_data, device):
    assert img_tensor.ndim == 4, "Image tensor should have 4 dimensions (batch_size, channels, height, width)"
    
    boxes = prediction_data.get("boxes", None)
    crops = []
    if boxes is None or img_tensor is None:
        return crops
    
    for box in boxes:
        if box is None:
            continue
        
        x1, y1, x2, y2 = box.int()
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_tensor.shape[-1], x2)
        y2 = min(img_tensor.shape[-2], y2)
        
        if x2 > x1 and y2 > y1:  # Valid crop
            crop = img_tensor[:, :, y1:y2, x1:x2].to(device)
            crops.append(crop)
    return crops
    
    
def get_cls_loss_target(prob, target_idx=None):
    target_tensor = torch.zeros_like(prob)
    
    if isinstance(target_idx, int):
        target_tensor[:, target_idx] = 1.0
    elif isinstance(target_idx, list):
        for i in target_idx:
            target_tensor[:, i] = 1.0
    elif target_idx is None:
        target_tensor = torch.ones_like(prob)
    return target_tensor


def object_detection_loss(pred_list, target_idx=None, device=None):
    if len(pred_list) == 0:
        return torch.tensor([0.0]).to(device), 0
    
    num_detections = 0
    subtotal_loss = 0.0
    for pred in pred_list:
        num_detections += len(pred["scores"])
        probs = pred["probs"]
        cls_loss_target = get_cls_loss_target(probs, target_idx=target_idx)
        cls_loss = 1.0 * F.mse_loss(probs, cls_loss_target, reduction='sum') / (len(probs) + 1)
        subtotal_loss += cls_loss
        
    return subtotal_loss, num_detections


def pipeline_infernce(args, model, processor, example, bx, device):
    prediction_list = []
    
    image_id, img_tensor, _ = object_detection_preprocess(example, processor, device=device)  
    
    bx = bx.to(img_tensor)
    img_tensor = bx + img_tensor 
    target_size = get_target_size(img_tensor)
    layer_1_prediction_data = object_detection_inference(
        model=model,
        processor=processor,
        img_tensor=img_tensor,
        target_size=target_size,
        thres=args.threshold,
    )
    prediction_list.append(layer_1_prediction_data)
    crops = cropper(img_tensor, layer_1_prediction_data, device)

    for i, cropped_tensor in enumerate(crops):        
        upsampled = F.interpolate(
            cropped_tensor, 
            size=(128, 128), 
            mode='bilinear', 
            align_corners=False,
        )
        
        target_size = get_target_size(upsampled)
        
        layer_2_prediction_data = object_detection_inference(
            model=model,
            processor=processor,
            img_tensor=upsampled,
            target_size=target_size,
            # thres=args.threshold,
            thres=0.5
        )
                
        if layer_2_prediction_data is None: # if a image is too small (h < 32 or w < 32), it will be rejected, hence prediction_data will be None
            continue
        
        prediction_list.append(layer_2_prediction_data)
        
    
    return bx, image_id, img_tensor, prediction_list


def pipeline_infernce_parallel(args, model, processor, example, bx, device):
    prediction_list = []
    
    image_id, img_tensor, _ = object_detection_preprocess(example, processor, device=device)
    bx = bx.to(img_tensor)
    img_tensor = bx + img_tensor
    target_size1 = get_target_size(img_tensor)  
    layer_1_prediction_data = object_detection_inference(
        model=model,
        processor=processor,
        img_tensor=img_tensor,
        target_size=target_size1,
        thres=args.threshold,
    )
    prediction_list.append(layer_1_prediction_data)
    crops = cropper(img_tensor, layer_1_prediction_data, device)  # list of tensors, each shape [1,3,h_i,w_i]
    
    
    if len(crops) == 0:
        return bx, image_id, img_tensor, prediction_list

    # Upsample each crop to (128×128).  Each `c` has shape [1,3,h_i,w_i], so after F.interpolate:
    #   up has shape [1,3,128,128].  We'll strip off the “batch‐dim = 1” so that we can stack along dim=0.
    upsampled_list = []
    for c in crops:
        up = F.interpolate(
            c,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )  # shape [1,3,128,128]
        up = up.squeeze(0)  # now shape [3,128,128]
        upsampled_list.append(up)

    batch_up = torch.stack(upsampled_list, dim=0)  # dtype and device match c

    N = batch_up.size(0)
    target_sizes2 = [torch.Size([128, 128])] * N
    raw2 = model(batch_up, target_sizes=target_sizes2)
    post2_list = processor.post_process_object_detection(
        raw2, target_sizes=target_sizes2, threshold=0.0
    )

    for i in range(N):
        outputs = post2_list[i]  # dict with "scores", "labels", "boxes"
        logits_i = raw2.logits[i]            # shape [num_queries, …]
        probs_i = torch.sigmoid(logits_i)    # still requires_grad, connected to batch_up[i]

        # combined_i = torch.cat(
        #     (
        #         outputs["boxes"],
        #         outputs["scores"].unsqueeze(1),
        #         probs_i[: len(outputs["scores"])],
        #     ),
        #     dim=1,
        # )

        layer_2_pred = {
            "scores": outputs["scores"],
            # "labels": outputs["labels"],
            "boxes": outputs["boxes"],
            # "logits": logits_i,
            "probs": probs_i,
            # "combined": combined_i,
        }
        prediction_list.append(layer_2_pred)

    return bx, image_id, img_tensor, prediction_list


def main(args, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    dataset = ms_coco_init(num_samples=args.num_examples)
    model, processor = object_detection_init(ckpt=0, num_q=args.num_queries, device=device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    for example in tqdm(dataset, desc="Processing COCO dataset"):
        log_dict = {}
        total_loss = 0.0
        bx = torch.zeros(1,3,640,640).requires_grad_(True)
        for it in range(args.it_num):
            
            _, img_id, img_tensor, prediction_list = pipeline_infernce(args, model, processor, example, bx, device)
            layer_1_prediction_data = prediction_list[:1]
            layer_2_prediction_data = prediction_list[1:]
            
            layer_1_loss, layer_1_num_dets = object_detection_loss(layer_1_prediction_data, 2, device)
            layer_2_loss, layer_2_num_dets = object_detection_loss(layer_2_prediction_data, 2, device)
            
            total_loss = layer_1_loss + layer_2_loss
            total_loss.backward()
            
            with torch.no_grad():
                bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
                bx.data = -1.5 * bx.grad+ bx.data
                bx.data.clamp_(-0.04, 0.04)  
                
                log_dict[f"iteration_{it}"] = {
                    "layer_1_num_dets": layer_1_num_dets,
                    "layer_1_loss": layer_1_loss.detach().cpu().item(),
                    "layer_2_num_dets": layer_2_num_dets,
                    "layer_2_loss": layer_2_loss.detach().cpu().item(),
                }
            
            # pdb.set_trace()
            torch.cuda.empty_cache()
            
        with open(os.path.join(args.save_dir, f"{img_id}.json"), 'w') as f:
            json.dump(log_dict, f, indent=4)
    
    
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
            
                    
def launch():
    world_size = torch.cuda.device_count()
    mp.spawn(parallel_main, args=(world_size, args), nprocs=world_size, join=True)
    
    
def parallel_main(rank, world_size, args):
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dataset = ms_coco_init(num_samples=args.num_examples)
        
    total_len = len(dataset)
    per_gpu_len = total_len // world_size
    start = rank * per_gpu_len
    end = total_len if rank == world_size - 1 else (rank + 1) * per_gpu_len
    sub_dataset = dataset.select(range(start, end))
    
    model, processor = object_detection_init(ckpt=0, num_q=args.num_queries, device=device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    for example in tqdm(sub_dataset, desc="Processing COCO dataset"):
        log_dict = {}
        total_loss = 0.0
        bx = torch.zeros(1,3,640,640).requires_grad_(True)
        for it in range(args.it_num):
            _, img_id, img_tensor, prediction_list = pipeline_infernce_parallel(args, model, processor, example, bx, device)
            layer_1_prediction_data = prediction_list[:1]
            layer_2_prediction_data = prediction_list[1:]
            
            layer_1_loss, layer_1_num_dets = object_detection_loss(layer_1_prediction_data, 2, device)
            layer_2_loss, layer_2_num_dets = object_detection_loss(layer_2_prediction_data, 2, device)
            
            total_loss = layer_1_loss + layer_2_loss
            total_loss = layer_1_loss
            total_loss.backward()
            
            with torch.no_grad():
                bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
                bx.data = -1.5 * bx.grad+ bx.data
                bx.data.clamp_(-0.04, 0.04)  
                
                log_dict[f"iteration_{it}"] = {
                    "layer_1_num_dets": layer_1_num_dets,
                    "layer_1_loss": layer_1_loss.detach().cpu().item(),
                    "layer_2_num_dets": layer_2_num_dets,
                    "layer_2_loss": layer_2_loss.detach().cpu().item(),
                }
            
            bx.grad.zero_()
            prediction_list.clear()
            
            torch.cuda.empty_cache()
        
        with open(os.path.join(args.save_dir, f"{img_id}.json"), 'w') as f:
            json.dump(log_dict, f, indent=4)
            
    dist.destroy_process_group()

if __name__ == "__main__":
    # Configuration
    TARGET = 2
    NUM_Q = 300
    IT_NUM = 200
    EPSILON = 0.04
    THRESHOLD = 0.5
    PARALLEL = False
    ITERATIONS = 200
    NUM_EXAMPLES = 1000
    formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = f"./saved/attack/{formatted_time}"
    
    argparser = argparse.ArgumentParser(description="Object Detection Pipeline")
    argparser.add_argument("--target", type=int, default=TARGET, help="Target class index for attack")
    argparser.add_argument("--threshold", type=float, default=THRESHOLD, help="Confidence threshold for predictions")
    argparser.add_argument("--num_queries", type=int, default=NUM_Q, help="Number of queries for the model")
    argparser.add_argument("--iterations", type=int, default=ITERATIONS, help="Number of iterations for the attack")
    argparser.add_argument("--epsilon", type=float, default=EPSILON, help="Epsilon value for perturbation")
    argparser.add_argument("--num_examples", type=int, default=NUM_EXAMPLES, help="Number of examples to process")
    argparser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Directory to save results")
    argparser.add_argument("--it_num", type=int, default=IT_NUM, help="Number of iterations per attack")
    argparser.add_argument("--parallel", action="store_true", default=PARALLEL, help="Run in parallel mode")
    
    args = argparser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # launch()
    if args.parallel:
        launch()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main(args, device=device)    