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


class odStream(Process):
    def __init__(self, img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device=None):
        super().__init__(name="ODStream")
        self.img2od_queue = img2od_queue
        self.od2fr_queue = od2fr_queue
        self.od2lpr_queue = od2lpr_queue
        self.od2cap_queue = od2cap_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
    
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("OD stream started")
            
            # Load model
            logger.info(f"Loading OD model {self.model_id}")
            self.model, self.processor = load_from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("OD Model loaded and ready")
            
            # Process images
            while not self.stop_event.is_set():
                try:
                    # Get next image with timeout
                    data = self.img2od_queue.get(timeout=2.0)
                    
                    # Check for end signal
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    request = self.denormalize(request)
                    request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        od_output = self.model(request)
                        
                    # Process detections
                    _output = self.processor.post_process_object_detection(
                        od_output,
                        threshold=0.25,
                        target_sizes=[request.shape[2:]]
                    )[0]
                    
                    # Extract results
                    scores, labels, boxes = _output["scores"], _output["labels"], _output["boxes"]
                    
                    """
                    ===============================================================================
                    from all bounding boxes, select ONLY "person" boxes for face recognition
                    """
                    found_valid_box = False
                    for label, box in zip(labels, boxes):
                        if int(label.item()) == 0: # person: 0
                            found_valid_box = True
                            x1, y1, x2, y2 = box
                            cropped_tensor = request[:, :, int(y1):int(y2), int(x1):int(x2)]
                            cropped_np = cropped_tensor.cpu().numpy()
                            self.od2fr_queue.put(cropped_np)
                            
                    if found_valid_box:
                        pass
                    else:
                        self.od2fr_queue.put(request.cpu().numpy())
                        
                    self.od2fr_queue.put("END OF FRAME")
                    """
                    ===============================================================================
                    from all bounding boxes, select ONLY "car" boxes for license plate recognition
                    """
                    found_valid_box = False
                    for label, box in zip(labels, boxes):
                        if int(label.item()) == 1: # car: 1
                            found_valid_box = True
                            x1, y1, x2, y2 = box
                            cropped_tensor = request[:, :, int(y1):int(y2), int(x1):int(x2)]
                            cropped_np = cropped_tensor.cpu().numpy()
                            self.od2lpr_queue.put(cropped_np)   

                    if found_valid_box:
                        pass
                    else:
                        self.od2lpr_queue.put(request.cpu().numpy())
                        
                    self.od2lpr_queue.put("END OF FRAME")
                    
                    """
                    ===============================================================================
                    from all bounding boxes, select traffic related boxes for image captioning
                    ["person", "car", "traffic lights", "stop sign", etc...]
                    """
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = float('-inf'), float('-inf')
                    found_valid_box = False
                    for label, box in zip(labels, boxes):
                        x1, y1, x2, y2 = box
                        if int(label.item()) in [0, 5, 1, 11, 2, 12, 9, 6, 7, 3]:
                            min_x = min(min_x, x1.item())
                            min_y = min(min_y, y1.item())
                            max_x = max(max_x, x2.item())
                            max_y = max(max_y, y2.item())
                            found_valid_box = True
                    if found_valid_box:
                        cropped_tensor = request[:, :, int(min_y):int(max_y), int(min_x):int(max_x)]
                        cropped_np = cropped_tensor.cpu().numpy()
                        self.od2cap_queue.put(cropped_np)
                    else:
                        self.od2cap_queue.put(request.cpu().numpy())
                        
                    # Clean up
                    del request, od_output, _output, scores, labels, boxes
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("img2od Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing img2od Queue: {str(e)}", exc_info=True)
            
            # Signal end to downstream processes
            for q in [self.od2fr_queue, self.od2lpr_queue, self.od2cap_queue]:
                q.put(None)
                
        except Exception as e:
            logger.error(f"Error in OD stream: {str(e)}", exc_info=True)
            # Signal end to downstream processes on error
            for q in [self.od2fr_queue, self.od2lpr_queue, self.od2cap_queue]:
                q.put(None)
        finally:
            logger.info("OD stream ended")
    
    def shutdown(self):
        self.stop_event.set()
        
        
    def denormalize(self, tensor):
        """
        Denormalizes a tensor using the provided mean and std.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
        return tensor