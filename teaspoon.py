import torch
from base_attack import BaseAttack
from transformers import DetrConfig, AutoImageProcessor, DetrForObjectDetection
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from PIL import Image
import requests
import torch.nn.functional as F
import pdb
import torch.nn as nn
import torchvision
import numpy as np
import time
from utils import set_all_seeds
set_all_seeds(0)
from model_zoo import load_from_pretrained
from torch.optim import Adam
import math

learning_rate = 0.001

class TeaSpoon(BaseAttack):
    
    def run_attack(self, image, img_id):
        self.init_input(image, img_id)
        self.generate_bx()
        self.adam_opt = Adam([self.bx], lr=learning_rate, amsgrad=True)
        
        for it in range(self.it_num):
            added_imgs = self.img_tensor + self.bx
            added_imgs.clamp_(min=0, max=1)
            self.inference(added_imgs)
            self.update_bx(it)
            self.logger(it)
            self.save_img_pt(added_imgs)
        self.write_log()
        
        self.clean_flag = True
        torch.cuda.empty_cache()

    def update_bx(self, it_count):
        width = self.boxes[:, 2] - self.boxes[:, 0]
        height = self.boxes[: ,3]- self.boxes[:, 1]
        sel_dets = self.scores
        sel_height = height.clone()
        sel_width = width.clone()

        sel_aaa = (sel_width/self.target_size[0][0]) * (sel_height/self.target_size[0][1])
        
        loss1_targets = torch.ones_like(sel_dets)
        loss1 = 10*(F.mse_loss(sel_dets, loss1_targets, reduction='sum'))
        
        loss2 = 40*torch.norm(self.bx, p=2)
        
        loss3 = 100*torch.sum(sel_aaa)
        
        cls_loss_target_tensor = self.cls_loss_target()
        loss4 = 0.1 * F.mse_loss(self.prob, cls_loss_target_tensor, reduction='sum')
        
        alpha1, alpha2, alpha3 = self.factor_scheduler(it_count)
        total_loss = alpha3 * loss2 + alpha2 * loss3 + alpha1 * loss1 + loss4
        total_loss.requires_grad_(True)
        
        self.adam_opt.zero_grad()
        total_loss.backward(retain_graph=True)
        self.bx.grad = self.bx.grad / (torch.norm(self.bx.grad,p=2) + 1e-20)
        self.bx.data = -1.5 * self.bx.grad+ self.bx.data
        
        self.clone_loss(loss1, loss2, loss3, loss4)
        return self.bx
        
    def cls_loss_target(self):
        target_tensor = torch.zeros_like(self.prob)
        if isinstance(self.target_idx, int):
            target_tensor[:, self.target_idx] = 1.0
            pdb.set_trace()
        elif isinstance(self.target_idx, list):
            for i in self.target_idx:
                target_tensor[:, i] = 1.0
        elif self.target_idx == None:
            target_tensor = self.prob.detach().clone()
        return target_tensor

    def factor_scheduler(self, it_count):
        alpha2 = 1 - math.cos(min(it_count / self.it_num * math.pi, math.pi / 2))
        alpha3 = 1 - math.cos(min(it_count / self.it_num * math.pi, math.pi / 2))
        alpha1 = 3 - alpha2 - alpha3
        return alpha1, alpha2, alpha3
    
    
    def clone_loss(self, loss1, loss2, loss3, loss4):
        self._loss1 = loss1.detach().clone().cpu().item()
        self._loss2 = loss2.detach().clone().cpu().item()
        self._loss3 = loss3.detach().clone().cpu().item()
        self._loss4 = loss4.detach().clone().cpu().item()
        
        
    def logger(self, it):
        count = 0
        if self.target_idx:
            for i in self.target_idx:
                count = count + (self.labels == i).sum().item()
        else:
            count = len(self.labels)
        
        tmp_dict = {
            "count" : count,
            "labels" : self.labels.tolist(),
            "scores" : self.scores.tolist(),
            "boxes" : self.boxes.tolist(),
            "time" : self.elapsed_time,
            "loss": [self._loss1, self._loss2, self._loss3, self._loss4]
            
        }
        
        self.result_dict[it] = tmp_dict
                
                
def single_test(num_q = 1000, 
                img_id = 0, 
                url = None,
                device = None):
    
    if not url:
        url = "https://farm5.staticflickr.com/4116/4827719363_31f75f0c8f_z.jpg"
    
    image = Image.open(requests.get(url, stream=True).raw)

    thres = 0.25
    
    for i in range(3):
        model, image_processor = load_from_pretrained(i, device)
        model.config.num_queries = num_q
        teaspoon = TeaSpoon(
            model = model,
            image_processor = image_processor,
            it_num = 100,
            conf_thres = thres,
            target_idx = [1,3],
            output_dir = f"./output_rt_detr/teaspoon/algo_{i}",
            device = device
        )
        teaspoon.run_attack(image, img_id)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    single_test(device=device)