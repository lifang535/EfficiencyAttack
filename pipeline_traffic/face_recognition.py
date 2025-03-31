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
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1 # https://github.com/timesler/facenet-pytorch/tree/master
from PIL import Image
import cv2
import torch.nn.functional as F


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class frStream(Process):
    def __init__(self, od2fr_queue, fr2kr_queue, device=None):
        super().__init__(name="FRStream")
        self.od2fr_queue = od2fr_queue
        self.fr2kr_queue = fr2kr_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = None  # Will be set later via set_config
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("FR stream started")
            
            logger.info(f"Loading FR model {self.model_id}")
            self.facenet = InceptionResnetV1(pretrained=f"{self.model_id}").eval().to(self.device)
            logger.info("FR Model loaded and ready")
                        
            while not self.stop_event.is_set():
                try:
                    # Non-blocking queue check
                    try:
                        data = self.od2fr_queue.get(timeout=2.0)
                    except Empty:
                        # Timeout occurred, check if we should exit
                        continue
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Process the data
                    request = torch.from_numpy(data).to(self.device)
                    request = self.facenet_padding(request)
                    
                    with torch.no_grad():
                        embeddings = self.facenet(request).cpu().numpy()
                    
                    self.fr2kr_queue.put(embeddings)
                    
                    # Explicit cleanup
                    del request, embeddings
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error processing od2fr Queue: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Fatal error in FR stream: {str(e)}", exc_info=True)
        finally:
            # Explicit cleanup of PyTorch resources
            if hasattr(self, 'facenet'):
                del self.facenet
            self.fr2kr_queue.put(None)  # Signal termination to the next process
            torch.cuda.empty_cache()
            logger.info("FR stream ended")
            self.shutdown()
            
            
    def shutdown(self):
        self.stop_event.set()
    
    def facenet_padding(self, image, min_size=160):
        # Get current dimensions
        if image.dim() == 4:
            image = image.squeeze(0)
            
        _, height, width = image.shape
        
        # Check if padding is needed
        if height < min_size or width < min_size:
            # Calculate padding
            pad_height = max(0, min_size - height)
            pad_width = max(0, min_size - width)
            
            # Calculate padding for each side
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # Apply padding
            image = F.pad(image.unsqueeze(0), 
                        (pad_left, pad_right, pad_top, pad_bottom), 
                        mode='constant', value=0)
            image = image.squeeze(0)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return image