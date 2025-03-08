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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class lmStream(Process):
    def __init__(self, cap2lm_queue, kr2lm_queue, device=None):
        super().__init__(name="LMStream")
        self.cap2lm_queue = cap2lm_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("LM stream started")
            
            logger.info(f"Loading LM model {self.model_id}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("LM Model loaded and ready")
            
            cap_end_received = False
            kr_end_received = False
            cap_data = None
            kr_data = None
            while not self.stop_event.is_set():
                # Exit condition: both upstream processes have terminated
                if cap_end_received and kr_end_received:
                    logger.info("Both CAP and KR streams have ended, terminating LM stream")
                    break
                
                try:
                    # Get data from CAP stream with short timeout
                    try:
                        cap_data = self.cap2lm_queue.get(timeout=0.5)
                        if cap_data is None:
                            cap_end_received = True
                            logger.info("Received end signal from CAP stream")
                        else:
                            # Process CAP data
                            pass
                    except Empty:
                        pass
                    
                    # Get data from KR stream with short timeout
                    try:
                        kr_data = self.kr2lm_queue.get(timeout=0.5)
                        if kr_data is None:
                            kr_end_received = True
                            logger.info("Received end signal from KR stream")
                        else:
                            # Process KR data
                            pass
                    except Empty:
                        pass
                    
                    # If we have both cap_data and kr_data, process them
                    if cap_data is not None and kr_data is not None and not isinstance(cap_data, bool) and not isinstance(kr_data, bool):
                        prompt = kr_data + cap_data
                        with torch.no_grad():
                            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                            generated_ids = self.model.generate(**encoded_input, 
                                                                max_new_tokens=50, 
                                                                do_sample=True,
                                                                pad_token_id=self.tokenizer.eos_token_id)                            
                            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Generated output: {decoded_text[:50]}...")
                        
                        del prompt, encoded_input, generated_ids, decoded_text
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Error processing in LM stream: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in LM stream: {str(e)}", exc_info=True)
        finally:
            logger.info("LM stream ended")
        
    def shutdown(self):
        self.stop_event.set()