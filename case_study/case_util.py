from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import datetime
import os
import argparse
import random
from PIL import Image
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import sys
import math
# Load model directly
sys.path.append("..")
sys.path.append("./CVPR22_NICGSlowDown")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        from datasets import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass

def ultra_fast_save(tensor, filepath):
    tensor_cpu = tensor.detach().cpu().contiguous()
    shape = tensor_cpu.shape

    shape_dims = np.array([len(shape)], dtype=np.int8).tobytes()
    shape_header = np.array(shape, dtype=np.int64).tobytes()

    dtype_str = np.dtype(tensor_cpu.numpy().dtype).str.encode('ascii')
    dtype_length = np.array([len(dtype_str)], dtype=np.int8).tobytes()

    data_bytes = tensor_cpu.numpy().tobytes()

    with open(filepath, 'wb') as f:
        fd = f.fileno()
        try:
            os.write(fd, shape_dims)
            os.write(fd, shape_header)
            os.write(fd, dtype_length)
            os.write(fd, dtype_str)
            os.write(fd, data_bytes)
            os.fsync(fd)
        except Exception as e:
            print(f"Failed to save tensor to {filepath}: {e}")
            os.remove(filepath)
                
def load_ms_coco_dataset(val_size):
    """Load MS COCO dataset with optional size limit"""
    from datasets import load_dataset
    coco_data = load_dataset("detection-datasets/coco", split="val")
    if val_size:
        random_indices = random.sample(range(len(coco_data)), val_size)
        return coco_data.select(random_indices)
    else:
        return coco_data

def load_rt_detr(model_id, num_q=1000, device=None):
    """Load RT-DETR model"""
    from model_zoo import load_from_pretrained
    model, processor = load_from_pretrained(model_id, num_q=num_q, device=device)
    model = model.eval()
    return model, processor

def load_git_base(device):
    """Load image captioning model"""
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained("microsoft/git-base", use_fast=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device)
    encoder = model.git.encoder
    tokenizer = processor.tokenizer
    # print(dir(model.git))
    return model, processor, encoder

def load_bclip(device):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model = model.eval()
    # print(dir(model))
    encoder = model.vision_model      
    decoder = model.text_decoder    
    
    return model, processor, encoder, decoder

def parse_example(data):
    """Parse example from COCO dataset"""
    image_id = data['image_id']
    image = data['image']
    width = data['width']
    height = data['height']
    bbox_id = data["objects"]["bbox_id"]
    category = data["objects"]["category"]
    bbox = data["objects"]["bbox"]
    area = data["objects"]["area"]
    
    return image_id, image, width, height, bbox_id, category, bbox, area

def denormalize(img_tensor):
    """Denormalize image tensor"""
    # Assuming the image has been normalized with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean

def merge_bbox(all_indices, boxes, device):
    # Get coordinates for all boxes
    all_boxes = boxes[all_indices]
    
    # Find the min/max coordinates to create a single large bounding box
    x_min = torch.min(all_boxes[:, 0])
    y_min = torch.min(all_boxes[:, 1])
    x_max = torch.max(all_boxes[:, 2])
    y_max = torch.max(all_boxes[:, 3])
    
    # Create merged bounding box
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, 640)
    y_max = min(y_max, 640)
    merged_box = torch.tensor([x_min, y_min, x_max, y_max]).to(device)

    return merged_box
    
def valid_bbox(box):
    try:
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            return False
        if ((x2 - x1) < 7) or ((y2 - y1) < 7): # ensure larger than kernel size
            return False
        return True
    except Exception as e:
        return False

def crop_box(data_tensor, box=None):
    try:
        if box is not None:
            x1, y1, x2, y2 = box
            height, width = data_tensor.shape[2], data_tensor.shape[3]
            cropped_tensor = data_tensor[:, :, int(y1):int(y2), int(x1):int(x2)].clone()
        else:
            cropped_tensor = data_tensor.clone()
        # cropped_np = cropped_tensor.cpu().numpy()
        return cropped_tensor
    except Exception as e:
        print(f"Error in cropping box: {str(e)}")
            
def inference(image_tensor, od_model, od_processor, ic_model, ic_processor, device):

    target_size = [image_tensor.shape[2:] for _ in range(1)]
    
    # Object detection
    od_output = od_model(image_tensor)
    od_pred = od_processor.post_process_object_detection(od_output, target_sizes=target_size, threshold=0.25)[0]
    _, labels, boxes = od_pred["scores"], od_pred["labels"], od_pred["boxes"]
    
    # Filter and merge bounding boxes
    face_indices = (labels == 0).nonzero(as_tuple=True)[0]
    plate_indices = (labels == 2).nonzero(as_tuple=True)[0]
    # print(f"face_indices: {face_indices}, plate_indices: {plate_indices}")
    if len(face_indices) > 0 or len(plate_indices) > 0:
        all_indices = torch.cat([face_indices, plate_indices]) if len(face_indices) > 0 and len(plate_indices) > 0 else face_indices if len(face_indices) > 0 else plate_indices
        merged_box = merge_bbox(all_indices, boxes, device)
        if valid_bbox(merged_box):
            cropped_tensor = crop_box(image_tensor, merged_box)  

            pixel_values = F.interpolate(cropped_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
            inputs = ic_processor(images=pixel_values, return_tensors="pt").to(device)
            prompt = "a photo of"
            text_inputs = ic_processor(text=prompt, return_tensors="pt").to(device)
            
            # First get the full model output to access logits
            ic_output = ic_model(
                pixel_values=inputs["pixel_values"],
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                return_dict=True,
                output_hidden_states=True,
            )
                        
            # Then generate the caption properly
            generation_output = ic_model.generate(
                pixel_values=inputs["pixel_values"].clone(),
                input_ids=text_inputs["input_ids"].clone(),
                attention_mask=text_inputs["attention_mask"].clone(),
                do_sample=True,
                top_p=0.9,              
                top_k=50,               
                max_new_tokens=512,      
                temperature=1,         
                num_beams=1,          
                min_length=10,          
                repetition_penalty=1,
                early_stopping=False,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            generated_ids = generation_output.sequences
            # scores = generation_output.scores
            caption = ic_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

            return (od_output, od_pred, caption, generation_output)
        else:
            return (od_output, od_pred)
    else:
        return (od_output, od_pred)
   
def train(image_tensor, od_model, od_processor, ic_model, ic_processor, device):

    target_size = [image_tensor.shape[2:] for _ in range(1)]
    od_output = od_model(image_tensor)
    od_pred = od_processor.post_process_object_detection(od_output, target_sizes=target_size, threshold=0.25)[0]
    cropped_tensor = image_tensor.clone()
    pixel_values = F.interpolate(cropped_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    inputs = ic_processor(images=pixel_values, return_tensors="pt").to(device)
    prompt = "a photo of"
    text_inputs = ic_processor(text=prompt, return_tensors="pt").to(device)
    ic_output = ic_model(
        pixel_values=inputs["pixel_values"],
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        return_dict=True,
        output_hidden_states=True,
    )

    generation_output = ic_model.generate(
        pixel_values=inputs["pixel_values"].clone(),
        input_ids=text_inputs["input_ids"].clone(),
        attention_mask=text_inputs["attention_mask"].clone(),
        do_sample=True,
        top_p=0.9,              
        top_k=50,               
        max_new_tokens=512,      
        temperature=1,         
        num_beams=1,          
        min_length=10,          
        repetition_penalty=1,
        early_stopping=False,
        output_scores=True,
        return_dict_in_generate=True,
        output_hidden_states=True
    )
    
    generated_ids = generation_output.sequences
    # scores = generation_output.scores
    caption = ic_processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    return (od_output, od_pred, caption, generation_output)
    
def od_loss_function(od_output, od_pred, target_size):
    
    logits = od_output.logits[0]
    prob = F.sigmoid(logits)
    cls_loss_target_tensor = torch.zeros_like(prob)
    cls_loss_target_tensor[:, 0] = 1.0
    cls_loss_target_tensor[:, 2] = 1.0
    loss_cls = F.mse_loss(prob, cls_loss_target_tensor, reduction='sum') / (len(logits) + 1)
    
    _, labels, boxes = od_pred["scores"], od_pred["labels"], od_pred["boxes"]
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[: ,3]- boxes[:, 1]
    sel_height = height.clone()
    sel_width = width.clone()
    sel_aaa = (sel_width/target_size[0][0]) * (sel_height/target_size[0][1])
    loss_area = torch.sum(sel_aaa) / (len(logits) + 1)
    return loss_cls, loss_area
    
def ic_loss_function(ic_output):
    logits = torch.cat(ic_output.scores, dim=0) # the logits here is in fact scores = softmax(logits)
    logits = logits + 1e-9
    hidden_states = ic_output.hidden_states
    # image_embeds = ic_output.image_embeds
    # last_hidden_state = ic_output.last_hidden_state
    
    processed_states = []
    for layer_idx in range(len(hidden_states)):
        for sublayer_idx in range(len(hidden_states[layer_idx])):
            tensor = hidden_states[layer_idx][sublayer_idx][0]
            if tensor.shape[0] == 1:
                processed_states.append(tensor)

    
    stacked_states = torch.stack(processed_states, dim=0).float()
    hidden_states = torch.abs(stacked_states)
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    loss_diversity = - torch.norm(hidden_states, p='nuc') 
    

    uni_distribution = (torch.ones(logits.shape) / logits.shape[1]).to(logits.device)
    loss_uncertainty = F.kl_div(torch.softmax(logits, dim=1), uni_distribution, reduction='sum') 
    
    logits = torch.softmax(logits, dim=1)
    eos_token_id = 102 # word_map['[SEP]'] == 102
    loss_eos = logits[:, eos_token_id].mean()
    
    # hidden_states = torch.stack(hidden_states, dim = 0).squeeze().float()   
    # hidden_states = torch.abs(hidden_states)
    # hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    # loss_diversity = - torch.norm(hidden_states, p='nuc') 

    return loss_uncertainty, loss_eos, loss_diversity
    
def clamp(delta, clean_imgs):
    clamp_imgs = (delta + clean_imgs).clamp(0, 1)
    clamp_delta = clamp_imgs - clean_imgs
    return clamp_delta
    
def run_attack(image_tensor, ic_model, od_model, ic_processor, od_processor, args, device):
    loss_name = ["cls loss", "area_loss", "uncertainty loss", "eos loss", "diversity loss"]
    delta = torch.randn_like(image_tensor, requires_grad=True)
    target_size = [image_tensor.shape[2:] for _ in range(1)]
    verbose_len, verbose_energy, verbose_latency = 0, 0, 0
    verbose_len_list, verbose_energy_list, verbose_latency_list, ori_latency_list, ori_energy_list, ori_len_list = [],[],[],[],[],[]

    tdx = 0
    for counter in range(args.iter):
        result = inference(image_tensor + delta, od_model, od_processor, ic_model, ic_processor, device)
        # result = train(image_tensor + delta, od_model, od_processor, ic_model, ic_processor, device)
        if len(result) == 4:
            od_output, od_pred, caption, ic_output = result
            loss_uncertainty, loss_eos, loss_diversity = ic_loss_function(ic_output)
            loss_cls, loss_area = od_loss_function(od_output, od_pred, target_size) 
            loss_cls_value = loss_cls.detach().clone()
            loss_area_value = loss_area.detach().clone()
            loss_uncertainty_value = loss_uncertainty.detach().clone()
            loss_eos_value = loss_eos.detach().clone()
            loss_diversity_value = loss_diversity.detach().clone()
            
            ratio1 = 10.0 * math.log(tdx + 1) - 20.0
            ratio2 = 0.5 * math.log(tdx + 1) + 1.0

            if tdx == 0:
                lambda1 = torch.abs(loss_uncertainty_value / loss_eos_value / ratio1)
                lambda2 = torch.abs(loss_uncertainty_value / loss_diversity_value / ratio2)
            else:
                cur_lambda1 = torch.abs(loss_uncertainty_value / loss_eos_value / ratio1)
                cur_lambda2 = torch.abs(loss_uncertainty_value / loss_diversity_value / ratio2)   
                # lambda1 = cur_lambda1
                # lambda2 = cur_lambda2                  
                lambda1 = 0.9 * last_lambda1 + 0.1 * cur_lambda1
                lambda2 = 0.9 * last_lambda2 + 0.1 * cur_lambda2
            last_lambda1, last_lambda2 = lambda1, lambda2  
            loss = loss_uncertainty + \
                   lambda1 * loss_eos + \
                   lambda2 * loss_diversity + \
                   loss_cls + \
                   loss_area
            od_loss = loss_cls + loss_area
            ic_loss = loss_uncertainty + lambda1 * loss_eos + lambda2 * loss_diversity
            loss = (od_loss * 500.0 + ic_loss * 8.0) / (500.0 + 8.0)
            # loss = loss_cls + loss_eos
            loss_list = [loss_cls_value.item(), loss_area_value.item(), loss_uncertainty_value.item(), loss_eos_value.item() * lambda1, loss_diversity_value.item() * lambda2]
            
            ic_model.zero_grad()
            od_model.zero_grad()
            
            loss.backward(retain_graph=False)   
            delta.data = delta - args.step_size * torch.sign(delta.grad.detach())
            # delta.grad = delta.grad / (torch.norm(delta.grad,p=2) + 1e-20)
            # delta.data = -1.5 * delta.grad + delta.data
            tdx += 1

        else:
            caption = "n/a"
            loss_cls, loss_area = od_loss_function(result[0], result[1], target_size)
            loss_cls_value = loss_cls.detach().clone()
            loss_area_value = loss_area.detach().clone()
            loss = loss_cls + loss_area
            loss_list = [loss_cls_value.item(), loss_area_value.item()]
            
            ic_model.zero_grad()
            od_model.zero_grad()
            
            loss.backward(retain_graph=False)  
            delta.grad = delta.grad / (torch.norm(delta.grad,p=2) + 1e-20)
            delta.data = -1.5 * delta.grad + delta.data
            
        delta.data = clamp(delta, image_tensor).clamp(-args.epsilon, args.epsilon)
        delta.grad.zero_()
        
        verbose_len_list.append(len(caption.split(' ')))
        if len(caption.split(' ')) > verbose_len:
            verbose_len = len(caption.split(' '))
            
        string = ""
        for i in range(len(loss_list)):
            string += f"{loss_name[i]:<15}: {loss_list[i]:4.2f} |"
        string += f"caption length: {len(caption.split(' ')):>4d} | "
        string += f"iter: {counter:>4d}"
        print(string)
        
    print(verbose_len)
    return image_tensor + delta, caption
    
def run_on_gpu(rank, args):
    device = torch.device(f'cuda:{rank}')
    
    od_model, od_processor = load_rt_detr(model_id=0, num_q=1000, device=device)
    ic_model, ic_processor, _, _ = load_bclip(device)

    total_data = load_ms_coco_dataset(val_size=args.total_size)
    local_data = total_data.shard(num_shards=args.world_size, index=rank)

    for data in tqdm(local_data, desc=f"GPU {rank}"):
        image_id, image, *_ = parse_example(data)
        image = image.convert("RGB")
        image = od_processor(image, return_tensors="pt")["pixel_values"].to(device)
        image_tensor = denormalize(image)

        adv_image, caption = run_attack(image_tensor, ic_model, od_model, ic_processor, od_processor, args, device)
        os.makedirs(f"./adv/gpu_{rank}", exist_ok=True)
        ultra_fast_save(adv_image, f"./adv/gpu_{rank}/img_{image_id}.pt")
    
set_seed(0) 
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack')
    parser.add_argument('--iter', default=1000, type=int)
    parser.add_argument('--step_size', default=0.0039, type=float)
    parser.add_argument('--epsilon', default=0.032 , type=float)
    parser.add_argument('--total_size', default=100, type=int)
    args = parser.parse_args()
    
    args.world_size = torch.cuda.device_count()
    mp.spawn(run_on_gpu, args=(args,), nprocs=args.world_size)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # od_model, od_processor = load_rt_detr(model_id=0, num_q=1000, device=device)    
    # ic_model, ic_processor, encoder, decoder = load_bclip(device)
    # resize_transform = transforms.Compose([transforms.Resize((224, 224)),])
    # coco_data = load_ms_coco_dataset(val_size=10)
    
    # word_map = ic_processor.tokenizer.get_vocab()

    # counter = 0
    # for data in tqdm(coco_data):
    #     image_id, image, width, height, bbox_id, category, bbox, area = parse_example(data)
    #     image = image.convert("RGB")
    #     image = od_processor(image, return_tensors="pt")["pixel_values"].to(device)
    #     image_tensor = denormalize(image)

    #     adv_image, caption = run_attack(image_tensor, ic_model, od_model, ic_processor, od_processor, args, device)
    #     ultra_fast_save(adv_image, f"./adv/img_{image_id}.pt")
                