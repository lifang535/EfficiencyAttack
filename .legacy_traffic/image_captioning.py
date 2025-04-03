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
        self.model_id = None
    
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("CAP stream started")
            
            logger.info(f"Loading CAP model {self.model_id}")
            # Initialize image transformation pipeline
            self.resize_transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])
            
            # Load the processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device) 
            self.model.eval()
            logger.info("CAP Model loaded and ready")
                        
            while not self.stop_event.is_set():
                try:
                    # Get data from the queue with timeout
                    data = self.od2cap_queue.get(block=False)0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Convert numpy array to tensor and move to device
                    request = torch.from_numpy(data).to(self.device)
                    
                    # Generate caption
                    with torch.no_grad():
                        caption = self.inference(request)
                        
                    # Send caption to language model queue
                    self.cap2lm_queue.put(str(caption))
                    
                    # Clean up resources
                    del request, caption
                    torch.cuda.empty_cache()
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2cap Queue: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in CAP stream: {str(e)}", exc_info=True)
        finally:
            # Always signal completion to the next process in the pipeline
            self.cap2lm_queue.put(None)
            logger.info("CAP stream ended")
            self.shutdown()
            
            
    def shutdown(self):
        """Signal the process to stop"""
        self.stop_event.set()
        
    def inference(self, image):
        """Generate a caption for the given image
        
        Args:
            image: PIL Image or torch.Tensor containing the image data
            
        Returns:
            str: Generated caption for the image
        """
        if isinstance(image, Image.Image):
            # Process PIL Image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values
        elif isinstance(image, torch.Tensor):
            # Process torch Tensor directly
            pixel_values = image.clone()
        else:
            raise TypeError("Input to the captioning model must be a PIL Image or a torch.Tensor")
        
        # Resize the image to the required dimensions
        pixel_values = self.resize_transform(pixel_values)
        
        # Generate caption
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption