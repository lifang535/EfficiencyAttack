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
sys.path.append("pytorch-licenseplate-segmentation")
import os
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model_zoo import load_from_pretrained
from pipeline_utils import  calculate_flops_decorator
from model import create_model # https://github.com/dbpprt/pytorch-licenseplate-segmentation/tree/master
from fast_plate_ocr import ONNXPlateRecognizer # https://github.com/ankandrew/fast-plate-ocr
import onnxruntime as ort
options = ort.SessionOptions()
from torchvision import transforms
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class lprStream(Process):
    def __init__(self, od2lpr_queue, lpr2kr_queue, device=None):
        super().__init__(name="LPRStream")
        self.od2lpr_queue = od2lpr_queue
        self.lpr2kr_queue = lpr2kr_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("LPR stream started")
            
            logger.info(f"Loading LPR model {self.model_id}")
            # self.model = ResNetForImageClassification.from_pretrained(self.model_id)
            # self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            # self.model.to(self.device)
            # self.model.eval()
            
            self.deeplabv3 = create_model()
            self.checkpoint = torch.load(self.model_id, map_location='cpu')
            self.deeplabv3.load_state_dict(self.checkpoint['model'])
            self.deeplabv3.eval().to(self.device) 
            
            self.onnx_lp_ocr = ONNXPlateRecognizer('argentinian-plates-cnn-model', 
                                                   providers=['CPUExecutionProvider'])
            logger.info("LPR Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2lpr_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    if isinstance(data, str) and data == "END OF FRAME":
                        self.lpr2kr_queue.put("END OF FRAME")
                        continue
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    # request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        # lp segment
                        output = self.pred(request, self.deeplabv3)
                        plate_tensor = self.post_process(output, request.detach().clone())
                        
                        # lp ocr
                        result = self.ocr(plate_tensor)
                        # print(result)
                        # raise ValueError("Stop here")
                    self.lpr2kr_queue.put(result[0])
                    
                    del request, output, plate_tensor, result
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("od2lpr Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2lpr Queue: {str(e)}", exc_info=True)
                    
            self.lpr2kr_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in LPR stream: {str(e)}", exc_info=True)
            self.lpr2kr_queue.put(None)
        finally:
            logger.info("LPR stream ended")
            
            
    def shutdown(self):
        self.stop_event.set()  


    def pred(self, image, model):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        output = model(image)['out'][0]
        return output
    
    
    def post_process(self, output, request, threshold=0.1):
        output = (output > threshold).type(torch.IntTensor)
        output = output.cpu().numpy()[0]
        result = np.where(output > 0)
        coords = list(zip(result[0], result[1]))
        if coords:
            y_coords, x_coords = zip(*coords)
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            if request.dim() == 4:
                request = request.squeeze(0)
            return torch.from_numpy(request[:, min_y:max_y+1, min_x:max_x+1])
        else:
            return request
            
            
    def ocr(self, tensor, save_dir=None):
        if save_dir is None:
            save_dir = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(save_dir, exist_ok=True)
        dest_dir = os.path.join(save_dir, "lpr.png")

        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
            np_image = tensor.cpu().numpy().transpose(1, 2, 0)
            if np_image.max() <= 1.0:
                np_image = (np_image * 255).astype(np.uint8)
            else:
                np_image = np_image.astype(np.uint8)
        pil_image = Image.fromarray(np_image)
        pil_image.save(dest_dir)
        result = self.onnx_lp_ocr.run(dest_dir)
        os.remove(dest_dir)
        os.rmdir(save_dir)
        return result
        

if __name__ == "__main__":
    model = create_model()
    checkpoint = torch.load("./pytorch-licenseplate-segmentation/model_v2.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    pass