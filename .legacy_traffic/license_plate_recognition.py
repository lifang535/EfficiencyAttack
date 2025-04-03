from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append("./")
sys.path.append("../")
sys.path.append("./pytorch-licenseplate-segmentation/")
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
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
        self.model_id = None
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("LPR stream started")
            
            logger.info(f"Loading LPR model {self.model_id}")
            
            # Load the license plate segmentation model
            self.deeplabv3 = create_model()
            self.checkpoint = torch.load(self.model_id, map_location='cpu')
            self.deeplabv3.load_state_dict(self.checkpoint['model'])
            self.deeplabv3.eval().to(self.device) 
            
            # Load OCR model
            self.onnx_lp_ocr = ONNXPlateRecognizer('argentinian-plates-cnn-model', 
                                                 providers=['CPUExecutionProvider'])
            logger.info("LPR Model loaded and ready")
                        
            while not self.stop_event.is_set():
                try:                    
                    data = self.od2lpr_queue.get(block=False)0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Convert numpy array to tensor and move to device
                    request = torch.from_numpy(data).to(self.device)
                    
                    with torch.no_grad():
                        # License plate segmentation
                        output = self.pred(request, self.deeplabv3)
                        plate_tensor = self.post_process(output, request.detach().clone())
                        
                        # License plate OCR
                        result = self.ocr(plate_tensor)
                    
                    self.lpr2kr_queue.put(result[0])
                    
                    # Clean up resources
                    del request, output, plate_tensor, result
                    torch.cuda.empty_cache()
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2lpr Queue: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in LPR stream: {str(e)}", exc_info=True)
        finally:
            self.lpr2kr_queue.put(None)  # Signal the next process
            logger.info("LPR stream ended")
            self.shutdown()
            
            
    def shutdown(self):
        self.stop_event.set()

    def pred(self, image, model):
        """Run prediction on the segmentation model"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        output = model(image)['out'][0]
        return output
    
    def post_process(self, output, request, threshold=0.1):
        """Extract the license plate region based on segmentation output"""
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
            
    def ocr(self, tensor):
        """Perform OCR on the license plate tensor"""
        # Create a temporary directory with timestamp
        save_dir = f"tmp_lpr_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        dest_path = os.path.join(save_dir, "lpr.png")

        # Convert tensor to image
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        np_image = tensor.cpu().numpy().transpose(1, 2, 0)
        
        # Normalize image values if needed
        if np_image.max() <= 1.0:
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = np_image.astype(np.uint8)
            
        # Save, process, and clean up
        pil_image = Image.fromarray(np_image)
        pil_image.save(dest_path)
        result = self.onnx_lp_ocr.run(dest_path)
        
        # Clean up temporary files
        os.remove(dest_path)
        os.rmdir(save_dir)
        
        return result
        

if __name__ == "__main__":
    model = create_model()
    checkpoint = torch.load("./pytorch-licenseplate-segmentation/model_v2.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    pass