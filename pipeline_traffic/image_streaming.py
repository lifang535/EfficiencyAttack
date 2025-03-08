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


class imgStream(Process):
    def __init__(self, img2od_queue, device=None):
        super().__init__(name="ImageStream")
        self.img2od_queue = img2od_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.folder_path = ""
        self.fps = 30
        
    def set_config(self, folder_path="", fps=30):
        self.folder_path = folder_path
        self.fps = fps
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("Image stream started")
            paths = sorted(glob.glob(f"{self.folder_path}/*.pt"))
            
            if not paths:
                logger.warning(f"No .pt files found in {self.folder_path}")
                self.img2od_queue.put(None)  # Signal end even if no files
                return
                
            logger.info(f"Found {len(paths)} image files")
            
            for p in tqdm(paths, desc="Processing images"):
                if self.stop_event.is_set():
                    break
                    
                # Load image to device memory
                image_tensor = torch.load(p, weights_only=True, map_location=self.device)
                
                # Convert to numpy array (safer for multiprocessing)
                numpy_array = image_tensor.cpu().numpy()
                
                # Send numpy array through queue
                self.img2od_queue.put(numpy_array)
                
                # Simulate frame rate
                time.sleep(1.0 / self.fps)
                
                # Clean up
                del image_tensor, numpy_array
                torch.cuda.empty_cache()
            
            # Signal end of stream
            logger.info("Image stream complete, sending end signal")
            self.img2od_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in image stream: {str(e)}", exc_info=True)
            self.img2od_queue.put(None)  # Make sure to signal end on error
        finally:
            logger.info("Image stream ended")
    
    def shutdown(self):
        self.stop_event.set()