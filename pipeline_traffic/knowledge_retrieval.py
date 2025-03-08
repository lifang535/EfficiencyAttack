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


class krStream(Process):
    def __init__(self, fr2kr_queue, lpr2kr_queue, kr2lm_queue, device=None):
        super().__init__(name="KRStream")
        self.fr2kr_queue = fr2kr_queue
        self.lpr2kr_queue = lpr2kr_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("KR stream started")
            
            logger.info(f"Loading KR model {self.model_id}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("KR Model loaded and ready")
            
            fr_data_list = []
            lpr_data_list = []
            fr_end_received = False
            lpr_end_received = False
            
            while not self.stop_event.is_set():
                try:
                    # Check if both upstream processes have terminated
                    if fr_end_received and lpr_end_received:
                        logger.info("Both FR and LPR streams have ended, terminating KR stream")
                        break
                        
                    # Process FR data
                    try:
                        fr_data = self.fr2kr_queue.get(timeout=1.0)
                        if fr_data is None:
                            fr_end_received = True
                            logger.info("Received end signal from FR stream")
                        elif isinstance(fr_data, str) and fr_data == "END OF FRAME":
                            pass  # Process frame boundary
                        else:
                            fr_data_list.append(fr_data)
                    except Empty:
                        pass
                        
                    # Process LPR data
                    try:
                        lpr_data = self.lpr2kr_queue.get(timeout=1.0)
                        if lpr_data is None:
                            lpr_end_received = True
                            logger.info("Received end signal from LPR stream")
                        elif isinstance(lpr_data, str) and lpr_data == "END OF FRAME":
                            pass  # Process frame boundary
                        else:
                            lpr_data_list.append(lpr_data)
                    except Empty:
                        pass
                        
                    # Process accumulated data if we have both LPR and FR data
                    if fr_data_list and lpr_data_list:
                        prompt = "".join(fr_data_list + lpr_data_list)
                        
                        with torch.no_grad():
                            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                            generated_ids = self.model.generate(**encoded_input, 
                                                                max_new_tokens=50, 
                                                                do_sample=True,
                                                                pad_token_id=self.tokenizer.eos_token_id)
                            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        
                        self.kr2lm_queue.put(decoded_text)
                        
                        del prompt, encoded_input, generated_ids, decoded_text
                        fr_data_list = []
                        lpr_data_list = []
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error in KR stream processing: {str(e)}", exc_info=True)
            
            # Signal end to LM stream
            logger.info("KR stream sending end signal to LM stream")
            self.kr2lm_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in KR stream: {str(e)}", exc_info=True)
            self.kr2lm_queue.put(None)
        finally:
            logger.info("KR stream ended")
            
    def shutdown(self):
        self.stop_event.set()