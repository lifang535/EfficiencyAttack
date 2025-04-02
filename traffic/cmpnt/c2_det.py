import sys
sys.path.append("../")
sys.path.append("../../")
from tqdm import tqdm
from multiprocessing import Process, Queue, Event
import torch
import glob
import time
from flops import FLOPs_DECORATOR, write_profile
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import multiprocessing as mp
import logging
from queue import Empty
from model_zoo import load_from_pretrained
from torch.profiler import profile, record_function, ProfilerActivity


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
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
        self.device = device
        
    def set_config(self, model_id, profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

    def shutdown(self):
        self.od2fr_queue.put(None)  
        self.od2lpr_queue.put(None) 
        self.od2cap_queue.put(None) 
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
            self.model, self.processor = load_from_pretrained(ckpt=self.model_id, 
                                                            num_q=1000, 
                                                            device=self.device)
            self.model.eval()
            logger.info(f"{self.__class__.__name__:<12} : OD model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                # Get next image with timeout
                
                try:
                    data = self.img2od_queue.get(timeout=2.0)
                    if data is None:  # End signal
                        break
                except Empty:
                    continue
                
                # data_tensor = torch.from_numpy(data).to(self.device)
                if isinstance(data, np.ndarray):
                    data_tensor = torch.from_numpy(data).to(self.device)
                elif isinstance(data, torch.Tensor):
                    data_tensor = data.to(self.device)
                    
                with torch.no_grad():
                    preds = self.model(data_tensor) 
                    output = self.processor.post_process_object_detection(
                        preds,
                        threshold=0.25,
                        target_sizes=[data_tensor.shape[2:]]
                    )[0]
                
                _, labels, boxes = output["scores"], output["labels"], output["boxes"]

                for label, box in zip(labels, boxes):
                    if int(label.item()) == 0:
                        cropped_np = self.crop_box(data_tensor, box)  
                        self.od2fr_queue.put(cropped_np)
                    
                    elif int(label.item()) == 2:
                        cropped_np = self.crop_box(data_tensor, box)  
                        self.od2lpr_queue.put(cropped_np)
                    
                cropped_np = self.crop_box(data_tensor)  
                self.od2cap_queue.put(cropped_np)
                        
                del cropped_np, data_tensor
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
            self.shutdown()

    def crop_box(self, data_tensor, box=None):
        try:
            if box is not None:
                x1, y1, x2, y2 = box
                height, width = data_tensor.shape[2], data_tensor.shape[3]
                cropped_tensor = data_tensor[:, :, int(y1):int(y2), int(x1):int(x2)].clone()
            else:
                cropped_tensor = data_tensor.clone()
            cropped_np = cropped_tensor.cpu().numpy()
            return cropped_np
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in cropping box: {str(e)}")
    
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


if __name__ == "__main__":
    pass