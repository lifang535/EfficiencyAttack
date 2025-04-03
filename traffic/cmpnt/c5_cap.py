from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
sys.path.append("../")
import logging
import os
from torchvision import transforms
from transformers import AutoModelForCausalLM # microsoft/git-base
from transformers import AutoProcessor
from flops import FLOPs_DECORATOR, write_profile, write_profile_lt
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)



class capStream(Process):
    def __init__(self, od2cap_queue, cap2lm_queue, device=None):
        super().__init__(name="CAPStream")
        self.od2cap_queue = od2cap_queue
        self.cap2lm_queue = cap2lm_queue
        self.stop_event = Event()
        self.device = device         
        self.model_id = None
    
    def set_config(self, model_id="microsoft/git-base", profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

    def run(self):    
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True,
                     profile_memory=False,
                     record_shapes=False
                     ) as prof:
            with record_function(f"{self.__class__.__name__}"):
                self._run()
        write_profile(prof, self.profile_save_path)
        
    # @FLOPs_DECORATOR
    def _run(self):
        try:
            self.resize_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
            
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device) 
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : CAP model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2cap_queue.get(block=False)
                    if data is None:  # End signal
                        break
                    
                    data_tensor = torch.from_numpy(data).to(self.device)
                    
                    with torch.no_grad():
                        caption = self.inference(data_tensor)
                        
                    while self.cap2lm_queue.full():
                        time.sleep(1)
                    self.cap2lm_queue.put(caption)
                    
                    torch.cuda.empty_cache()
                    
                except Empty:
                    continue
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
            self.shutdown()

    def shutdown(self):
        while self.cap2lm_queue.full():
            time.sleep(1)
        self.cap2lm_queue.put(None)
        self.stop_event.set()
        
        try:
            if hasattr(self, 'model'):
                try:
                    self.model = self.model.to("cpu")
                except:
                    pass
                del self.model
                self.model = None
            if hasattr(self, 'processor'):
                del self.processor
                self.processor = None
                
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in shutting down {str(e)}")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown successful")
            
    def inference(self, data_tensor):
        if isinstance(data_tensor, Image.Image):
            # Process PIL Image
            inputs = self.processor(images=data_tensor, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values
        elif isinstance(data_tensor, torch.Tensor):
            # Process torch Tensor directly
            pixel_values = data_tensor.clone()
        else:
            raise TypeError("Input to the captioning model must be a PIL Image or a torch.Tensor")
        
        pixel_values = self.resize_transform(pixel_values)
        
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
        