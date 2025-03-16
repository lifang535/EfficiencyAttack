from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append("../")
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model_zoo import load_from_pretrained
from pipeline_utils import  calculate_flops_decorator
import requests
from dotenv import load_dotenv

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

        self.kr_buffer = []  # Knowledge retrieval buffer (max size: 100)
        self.cap_buffer = []  # Captioning buffer (max size: 10)
        
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
            
            # cap_data = None
            # kr_data = None
            buffer_flag = False
            while not self.stop_event.is_set():
                # Exit condition: both upstream processes have terminated
                if cap_end_received and kr_end_received:
                    logger.info("Both CAP and KR streams have ended, terminating LM stream")
                    break
                
                # read, update cap buffer
                try:
                    if not cap_end_received:
                        cap_data = self.cap2lm_queue.get(timeout=0.5)
                        if cap_data is None:
                            cap_end_received = True
                            logger.info("Received end signal from CAP stream")
                        else:
                            self.cap_buffer.append(cap_data)
                            if len(self.cap_buffer) > 10:  # max size 10
                                self.cap_buffer.pop(0)
                            buffer_flag = True # buffer is updated
                except Empty:
                    pass

                # read, update kr buffer
                try:
                    if not kr_end_received:
                        kr_data = self.kr2lm_queue.get(timeout=0.5)
                        if kr_data is None:
                            kr_end_received = True
                            logger.info("Received end signal from KR stream")
                        else:
                            self.kr_buffer.append(kr_data)
                            if len(self.kr_buffer) > 100:  # max size 100
                                self.kr_buffer.pop(0)
                            buffer_flag = True # buffer is updated
                except Empty:
                    pass
                
                if buffer_flag: # do not do inference when no new info
                    prompt = ""
                    if self.kr_buffer:
                        prompt += " ".join(self.kr_buffer[-5:])  # use latest 5 
                    if self.cap_buffer:
                        prompt += " " + " ".join(self.cap_buffer[-2:])  # use latest 2

                    if prompt:
                        with torch.no_grad():
                            # use grok 2
                            # self.grok_chat_completion(prompt, stream=False, temperature=0, max_tokens=50)

                            # use GPT-2
                            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                            generated_ids = self.model.generate(
                                **encoded_input, 
                                max_new_tokens=50, 
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )   
                            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Generated output: {decoded_text[:50]}...")

                        del encoded_input, generated_ids, decoded_text
                        torch.cuda.empty_cache()
                        buffer_flag = False # toggle flag
                # try:
                #     # Get data from CAP stream with short timeout
                #     try:
                #         cap_data = self.cap2lm_queue.get(timeout=0.5)
                #         if cap_data is None:
                #             cap_end_received = True
                #             logger.info("Received end signal from CAP stream")
                #         elif not cap_end_received:
                #             with torch.no_grad():
                #                 prompt = cap_data
                #                 encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                #                 generated_ids = self.model.generate(**encoded_input, 
                #                                                     max_new_tokens=50, 
                #                                                     do_sample=True,
                #                                                     pad_token_id=self.tokenizer.eos_token_id)   
                #                 decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                                
                #                 # self.grok_chat_completion(prompt, stream=False, temperature=0, max_tokens=50)
                #                 # logger.info(f"Generated output: {decoded_text[:50]}...")
                            
                #             del prompt, encoded_input, generated_ids, decoded_text
                #             torch.cuda.empty_cache()
                #             pass
                #             pass
                #     except Empty:
                #         pass
                    
                #     # Get data from KR stream with short timeout
                #     try:
                #         kr_data = self.kr2lm_queue.get(timeout=0.5)
                #         if kr_data is None:
                #             kr_end_received = True
                #             logger.info("Received end signal from KR stream")
                #         elif not kr_end_received:
                #             with torch.no_grad():
                #                 prompt = kr_data
                #                 encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                #                 generated_ids = self.model.generate(**encoded_input, 
                #                                                     max_new_tokens=50, 
                #                                                     do_sample=True,
                #                                                     pad_token_id=self.tokenizer.eos_token_id)   
                #                 decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                                
                #                 # self.grok_chat_completion(prompt, stream=False, temperature=0, max_tokens=50)
                #                 # logger.info(f"Generated output: {decoded_text[:50]}...")
                            
                #             del prompt, encoded_input, generated_ids, decoded_text
                #             torch.cuda.empty_cache()
                #             pass
                #     except Empty:
                #         pass
                    
                    # If we have both cap_data and kr_data, process them
                    # if cap_data is not None and kr_data is not None and not isinstance(cap_data, bool) and not isinstance(kr_data, bool):
                    #     prompt = kr_data + cap_data
                    #     with torch.no_grad():
                            
                    #         encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    #         generated_ids = self.model.generate(**encoded_input, 
                    #                                             max_new_tokens=50, 
                    #                                             do_sample=True,
                    #                                             pad_token_id=self.tokenizer.eos_token_id)   
                    #         decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            
                    #         # self.grok_chat_completion(prompt, stream=False, temperature=0, max_tokens=50)
                    #         # logger.info(f"Generated output: {decoded_text[:50]}...")
                        
                    #     del prompt, encoded_input, generated_ids, decoded_text
                    #     torch.cuda.empty_cache()
                
                # except Exception as e:
                #     logger.error(f"Error processing in LM stream: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in LM stream: {str(e)}", exc_info=True)
        finally:
            logger.info("LM stream ended")
        
    def shutdown(self):
        self.stop_event.set()
        

    def grok_chat_completion(self, content, model="grok-2-latest", stream=False, temperature=0, max_tokens=50):
        load_dotenv()
        api_key = os.getenv("GROK_2_API_KEY")
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "messages": [
                # {
                #     "role": "system",
                #     "content": "You are a test assistant."
                # },
                {
                    "role": "user",
                    "content": f"{content}"
                }
            ],
            "model": f"{model}",
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens 
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
        
if __name__ == "__main__":
    pass