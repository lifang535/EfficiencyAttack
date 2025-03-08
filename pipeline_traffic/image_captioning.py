"""

                                  ___ face recognition _________
                                 /                              \______ knowledge ___
                                /                               /       retrieval    \ 
data ----- object detection ---|----- license plate recognition                       \     
                                \                                                      |--- language model
                                 \                                                    /
                                  \___ image captioning _____________________________/


object detection:
    - YOLO or Vision Transformer
    - Input: PIL Image
    - Output: (box, cls, scores)
    
face recognition:
    - ResNet 101
    - Input: bounding boxes which has a cls label == "person"
    - Output: 
    
license plate recognition:
    - ResNet 18
    - Input: bounding boxes which has a cls label == "car"
    - Output:
    
image captioning:
    - ResNet 101
    - Input: bounding boxes which has a cls label == ["person", "car", "traffic lights", "stop sign"]
    - Output:
        
language model:
    - GPT 2
    
"""


from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
sys.path.append("../")
import os
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model_zoo import load_from_pretrained
from pipeline_utils import  calculate_flops_decorator
from PIL import Image
from transformers import AutoModelForCausalLM # microsoft/git-base
from transformers import AutoProcessor
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class capStream(Process):
    def __init__(self, od2cap_queue, cap2lm_queue, device=None):
        super().__init__(name="CAPStream")
        self.od2cap_queue = od2cap_queue
        self.cap2lm_queue = cap2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("CAP stream started")
            
            logger.info(f"Loading CAP model {self.model_id}")
            # self.model = ResNetForImageClassification.from_pretrained(self.model_id)
            # self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            # self.model.to(self.device)
            # self.model.eval()
            
            self.resize_transform = transforms.Compose([transforms.Resize((224, 224)),])
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device) 
            self.model.eval()
            logger.info("CAP Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2cap_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    # request = request.unsqueeze(0)  
                    
                    with torch.no_grad():
                        # logits = self.model(request).logits
                        # caption = logits.argmax(-1).item()
                        caption = self.inference(request)
                        
                    self.cap2lm_queue.put(str(caption))
                    
                    del request, caption
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("od2cap Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2cap Queue: {str(e)}", exc_info=True)
                    
            self.cap2lm_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in CAP stream: {str(e)}", exc_info=True)
            self.cap2lm_queue.put(None)
        finally:
            logger.info("CAP stream ended")
            
    def shutdown(self):
        self.stop_event.set()
        
    def inference(self, image):
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values.to(self.device)
        elif isinstance(image, torch.Tensor): 
            pixel_values = image.clone()
        else:
            raise TypeError("input of the ms-captioning model has to be an PIL Image or a torch.Tensor")
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = self.resize_transform(pixel_values)
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption