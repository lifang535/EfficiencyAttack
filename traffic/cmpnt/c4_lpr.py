from multiprocessing import Process, Event, Queue
from queue import Empty
import multiprocessing as mp
import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import os
import glob
import logging
import sys
import time
from PIL import Image
sys.path.append("../")
from flops import FLOPs_DECORATOR, write_profile
from torch.profiler import profile, record_function, ProfilerActivity
# from model import create_model # https://github.com/dbpprt/pytorch-licenseplate-segmentation/tree/master
from fast_plate_ocr import ONNXPlateRecognizer # https://github.com/ankandrew/fast-plate-ocr
import onnxruntime as ort
options = ort.SessionOptions()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
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
        
    def set_config(self, model_id="cmpnt/model_v2.pth", profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

    def shutdown(self):
        self.lpr2kr_queue.put(None)
        self.stop_event.set()
        
    def run(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True,
                     profile_memory=True,
                     record_shapes=True
                     ) as prof:
            with record_function("lmStream"):
                self._run()
                
        write_profile(prof, self.profile_save_path)
        
    # @FLOPs_DECORATOR
    def _run(self):
        try:
            self.deeplabv3 = self.create_model()
            self.checkpoint = torch.load(self.model_id, map_location='cpu')
            self.deeplabv3.load_state_dict(self.checkpoint['model'])
            self.deeplabv3.eval().to(self.device) 
            self.onnx_lp_ocr = ONNXPlateRecognizer('argentinian-plates-cnn-model', 
                                                    providers=['CPUExecutionProvider'])
            logger.info(f"{self.__class__.__name__:<12} : LPR model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                try:                    
                    data = self.od2lpr_queue.get(timeout=2.0)
                    if data is None:
                        logger.info(f"{self.__class__.__name__:<12} : Received end signal")
                        break
                    
                    data_tensor = torch.from_numpy(data).to(self.device)
                    
                    # Run the segmentation model
                    with torch.no_grad():
                        pred = self.segmentation(data_tensor, self.deeplabv3)
                        plate_tensor = self.post_process(pred, data_tensor.detach().clone())
                        plate_text = self.ocr(plate_tensor)[0]
                    
                    # Send the result to the next queue
                    self.lpr2kr_queue.put(plate_text)
                    
                    del data_tensor, pred, plate_tensor, plate_text
                    torch.cuda.empty_cache()
                
                except Empty:
                    continue
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
            self.shutdown()

    def segmentation(self, image, model):
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
            
    def ocr(self, plate_tensor):
        uint8_tensor = plate_tensor.type(torch.uint8)
        if uint8_tensor.dim() == 4:
            uint8_tensor.squeeze_(0)
        
        gray_tensor = 0.299 * uint8_tensor[0] + 0.587 * uint8_tensor[1] + 0.114 * uint8_tensor[2]
        gray_tensor = gray_tensor.to(torch.uint8)  # Convert back to uint8 after grayscale calculation
        gray_tensor.unsqueeze_(-1)
            
        plate_array = gray_tensor.cpu().numpy()
        
        result = self.onnx_lp_ocr.run(plate_array)
        return result
    
    def create_model(self, outputchannels=1, aux_loss=True, freeze_backbone=False):
        model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
            progress=True, 
            aux_loss=aux_loss)

        if freeze_backbone is True:
            for p in model.parameters():
                p.requires_grad = False

        model.classifier = DeepLabHead(2048, outputchannels)

        return model