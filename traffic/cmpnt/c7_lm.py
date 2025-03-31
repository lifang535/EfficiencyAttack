from multiprocessing import Process, Event
import sys
sys.path.append("../")
import time
import torch
import logging
from queue import Empty
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
from dotenv import load_dotenv
import os
import multiprocessing as mp
from flops import FLOPs_DECORATOR
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class lmStream(Process):
    def __init__(self, cap2lm_queue, kr2lm_queue, device=None):
        super().__init__(name="LMStream")
        self.cap2lm_queue = cap2lm_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = None

        # Buffers for storing recent data
        self.kr_buffer = []  # Knowledge retrieval buffer (max size: 100)
        self.cap_buffer = []  # Captioning buffer (max size: 10)
        
    def set_config(self, model_id="gpt2"):
        self.model_id = model_id
        
    def shutdown(self):
        logger.info(f"{self.__class__.__name__:<12} : shutting down")
        self.stop_event.set()
        
    @FLOPs_DECORATOR
    def run(self):
        try:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : LM model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            cap_end_received = False
            kr_end_received = False
            
            while not self.stop_event.is_set():
                if cap_end_received and kr_end_received:
                    break

                # Drain captioning queue
                if not cap_end_received:
                    try:
                        data = self.cap2lm_queue.get(timeout=0.1)
                        if data is None:
                            cap_end_received = True
                            logger.info(f"{self.__class__.__name__:<12} : Received end signal from CAP")
                        else:
                            pass
                    except Empty:
                        continue
                    
                if not kr_end_received:
                    try:
                        data = self.kr2lm_queue.get(timeout=0.1)
                        if data is None:
                            kr_end_received = True
                            logger.info(f"{self.__class__.__name__:<12} : Received end signal from KR")
                        else:
                            pass
                    except Empty:
                        continue
                    
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received BOTH END signal, shutting down")
            self.shutdown()
            
            
    def grok_chat_completion(self, content, model="grok-2-latest", stream=False, temperature=0, max_tokens=50):
        """Make an API call to the Grok-2 API"""
        load_dotenv()
        api_key = os.getenv("GROK_2_API_KEY")
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "messages": [{"role": "user", "content": content}],
            "model": model,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens 
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
                    
                    