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

# Append necessary paths
sys.path.append("../")
sys.path.append(str(Path(__file__).resolve().parent.parent))


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
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    def shutdown(self):
        """Signal the process to stop"""
        self.stop_event.set()
    
    def _drain_queue(self, queue, buffer, max_size, timeout=0.1):
        """
        Drains available items from the queue.
        Returns a tuple (buffer_updated, end_signal_received)
        """
        updated = False
        end_signal = False
        try:
            item = queue.get(timeout=timeout)
            if item is None:
                logger.info("Received end signal from queue")
                end_signal = True
            else:
                buffer.append(item)
                if len(buffer) > max_size:
                    buffer.pop(0)
                updated = True
            # Drain any remaining items
            while True:
                try:
                    item = queue.get_nowait()
                    if item is None:
                        logger.info("Received end signal from queue")
                        end_signal = True
                    else:
                        buffer.append(item)
                        if len(buffer) > max_size:
                            buffer.pop(0)
                        updated = True
                except Empty:
                    break
        except Empty:
            pass
        return updated, end_signal

    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("LM stream started")
            if not self.model_id:
                raise ValueError("Model ID not set. Please call set_config with a valid model ID.")
            
            logger.info(f"Loading LM model {self.model_id}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("LM Model loaded and ready")
            
            cap_end_received = False
            kr_end_received = False
            
            while not self.stop_event.is_set():
                # Drain both queues and update buffers
                cap_updated, cap_end = self._drain_queue(self.cap2lm_queue, self.cap_buffer, 10)
                kr_updated, kr_end = self._drain_queue(self.kr2lm_queue, self.kr_buffer, 100)
                if cap_end:
                    cap_end_received = True
                    # logger.info("CAP stream ended")
                if kr_end:
                    kr_end_received = True
                    # logger.info("KR stream ended")
                
                # Terminate if both upstream streams have ended
                if cap_end_received and kr_end_received:
                    logger.info("Both CAP and KR streams have ended, terminating LM stream")
                    break
                
                # Generate output if any new data is received
                if cap_updated or kr_updated:
                    self.generate_from_buffers()
                
                # Cleanup GPU cache
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error in LM stream: {e}", exc_info=True)
        finally:
            logger.info("LM stream ended")
            self.shutdown()
            
        
    def generate_from_buffers(self):
        """Generate text from the current buffer contents"""
        # Build prompt from the last 5 KR items and last 2 CAP items
        prompt_parts = []
        if self.kr_buffer:
            prompt_parts.append(" ".join(self.kr_buffer[-5:]))
        if self.cap_buffer:
            prompt_parts.append(" ".join(self.cap_buffer[-2:]))
        prompt = " ".join(prompt_parts)
            
        if prompt:
            with torch.no_grad():
                encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(
                    **encoded_input, 
                    max_new_tokens=50, 
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                # logger.info(f"Generated output: {decoded_text[:50]}...")
                torch.cuda.empty_cache()

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
        
if __name__ == "__main__":
    pass