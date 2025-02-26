
"""
Multi-layer DyDNN System (MLDDS)

Based on what we have described previously in this paper, a MLDDS is typically built by organizing 
a group of open-sourced, publicly available models. Among all those models, we noticed that most of
them adapt a similiar manner of training and architecture with some popular DNN backbones, such as
ResNet, GPT and VGG etc.. Due to this reason and it is actually impossible and no need to list all 
possible design of a MLDDS, we can say that a typical MLDDS is constructed using ResNet-based, 
VGG-based and GPT-based models.


Backbones:

- ResNet
- GPT
- VGG

Generally speaking, ResNet-based models are considered to be non-dynamic(static computational cost),
while DyDNNs can adapt GPT-based and VGG-based backbones such as early-exit models. To be more 
speicifc, GPT-like models use a decoder-only architecture for text generation task, the length of 
output text is dynamic and dependent on the input, until reach the max length or stop at the EOS; 
similarily, there are many early-exit models based on VGG backbones, which means that the inference
pathway of the model is dependent on the input, so does the computational cost.

In this paper, GPT reperensents a dynamic DNN that process text and VGG represents a dynamic DNN 
that process images and video frames.

Literature:
- survey of early exiting [https://arxiv.org/pdf/2103.04505v4]
"""

import cv2
import glob
import torch
import numpy as np
import time
import functools
from transformers import GPT2Tokenizer, GPT2Model
import timm
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
import pdb
from model_zoo import load_from_pretrained

sample_path = "./sample_data/sample.png"
sample_image = Image.open(sample_path).convert("RGB")
sample_prompt = "how many 'r's in the word strawberry?"

class BaseApp:
    def __init__(self, device=None):
        self.device = device
        pass

    def load_models(self):
        
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._gpt2 = GPT2Model.from_pretrained('gpt2')
        
        self._resnet152_image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-152", use_fast=True)
        self._resnet152 = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")
        
        self._resnet101_image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-101", use_fast=True)
        self._resnet101 = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")
        
        self._resnet50_image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", use_fast=True)
        self._resnet50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        
        self._resnet18_image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18", use_fast=True)
        self._resnet18 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
        
    def resnet_inference(self, resnet_model, resnet_image_proc, image):
        if image is None:
            image = sample_image
            
        if isinstance(image, Image.Image):
            inputs = resnet_image_proc(image, return_tensors="pt").to(self.device)
        elif isinstance(image, torch.tensor):
            inputs = image.clone().to(self.device)
        else:
            raise TypeError("invalid datatype of image")
        
        with torch.no_grad():
            logits = resnet_model(**inputs).logits
            predicted_label = logits.argmax(-1).item()
        return predicted_label
    
    def gpt_inference(self, gpt_model, gpt_tokenizer, text):
        if text is None or not isinstance(text, str):
            text = sample_prompt
        with torch.no_grad():
            encoded_input = gpt_tokenizer(text, return_tensors='pt')
            output = gpt_model(**encoded_input)
        return output
    

    def crop_image_tensor():
        pass
    
    

            
            # inference object detection model here
"""

                      ___ face recognition _______________
                     /                                    \   
                    /                                      \ 
object detection --|----- license plate recognition -------|--- language model
                    \                                     /  
                     \                                   /
                      \___ image captioning ____________/


object detection:
    - YOLO or Vision Transformer
    
face recognition:
    - ResNet 101
    
license plate recognition:
    - ResNet 18
    
image captioning:
    - ResNet 101
    
language model:
    - GPT 2
    

"""
if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(sample_image)
    torch.save(image_tensor, "./sample_data/sample.pt")
    pass