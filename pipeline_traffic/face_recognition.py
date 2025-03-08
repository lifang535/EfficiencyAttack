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
    - FaceNet
    - Input: bounding boxes which has a cls label == "person"
    - Output: 
    
license plate recognition:
    - DeepLab V3
    - Input: bounding boxes which has a cls label == "car"
    - Output:
    
image captioning:
    - microsoft/git-base
    - Input: bounding boxes which has a cls label == ["person", "car", "traffic lights", "stop sign"]
    - Output:
        
knowledge retrieval:
    - database of knowledge
    - Input: query
    - Output: knowledge
    
language model:
    - GPT 2/Grok 2
    
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
from pipeline_utils import calculate_flops_decorator
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1 # https://github.com/timesler/facenet-pytorch/tree/master
from PIL import Image

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
        
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("FR stream started")
            
            logger.info(f"Loading FR model {self.model_id}")
            # self.model = ResNetForImageClassification.from_pretrained(self.model_id)
            # self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            # self.model.to(self.device)
            # self.model.eval()
            

            # Create an inception resnet (in eval mode):
            self.facenet = InceptionResnetV1(pretrained=f"{self.model_id}").eval().to(self.device) # vggface2
            self.facenet.classify = True
            
            logger.info("FR Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2fr_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    if isinstance(data, str) and data == "END OF FRAME":
                        self.fr2kr_queue.put("END OF FRAME")
                        continue
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    # request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        # to_pil = transforms.ToPILImage()
                        # pil_image = to_pil(request[0])
                        resnet_o = self.facenet(request)
                        predicted_label = resnet_o[0].argmax()
                    
                    self.fr2kr_queue.put(str(predicted_label))
                    
                    del request, resnet_o, predicted_label
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("od2fr Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2fr Queue: {str(e)}", exc_info=True)
                    
            self.fr2kr_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in FR stream: {str(e)}", exc_info=True)
            self.fr2kr_queue.put(None)
        finally:
            logger.info("FR stream ended")
            
    def shutdown(self):
        self.stop_event.set()
        
        
