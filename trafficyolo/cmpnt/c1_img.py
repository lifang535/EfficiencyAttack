from pathlib import Path
import torch
from torchvision.io import read_image          # built‑in PNG/JPEG/PNG reader

import sys
sys.path.append("../")
from tqdm import tqdm
from multiprocessing import Process, Queue, Event
import torch
import glob
import time
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import multiprocessing as mp
import logging
import random
import gc


import pynvml
import json

random.seed(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_gaussian_noise(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

class imgStream(Process):
    def __init__(self, img2od_queue, device=None):
        super().__init__(name="ImageStream")
        self.img2od_queue = img2od_queue
        self.stop_event = Event()
        self.device = device
        
    def set_config(self, src_folder_path = None, fps = 30, profile_save_path = None, eval_size = 50):
        os.makedirs(profile_save_path, exist_ok=True)
        self.src_folder_path = src_folder_path
        self.fps = fps
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
        self.eval_size = eval_size
        
    def enable_profile(self, flag):
        self.flag = flag
        logger.info(f"{self.__class__.__name__:<12} : internal profiling set to {self.flag}")


    def ultra_fast_load(self, tmp_p):
        png_path = Path(tmp_p)
        try:
            img = read_image(str(png_path)).float() / 255.0  # Normalize to [0, 1]
            if img.ndim == 3:
                img = img.unsqueeze(0)  # Add batch dimension
            elif img.ndim != 4:
                raise ValueError(f"Unexpected image dimensions: {img.shape}")
            return img.numpy()
        except Exception as e:
            print(f"Failed to load image at {tmp_p}: {e}")
            return None

    def run(self):
        if self.flag:
            self.profile_run()
        else:
            self._run()
            
    def shutdown(self):
        while self.img2od_queue.full():
            time.sleep(0.01)
        self.img2od_queue.put(None)
        # self.img2od_queue.close()
        self.stop_event.set()
        
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    # torch.cuda.synchronize()
                except:
                    pass
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in shutting down {str(e)}")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown successful")
   
    def profile_run(self):    
        from torch.profiler import profile, record_function, ProfilerActivity
        from torch.profiler import schedule
        # torch.cuda.synchronize()   
        torch.cuda.empty_cache() 
        gc.collect()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True,
                     profile_memory=False,
                     record_shapes=False
                     ) as prof:
            with record_function(f"{self.__class__.__name__}"):
                # torch.cuda.synchronize()
                self._run()
                # torch.cuda.synchronize()   
        # torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        write_profile(prof, self.profile_save_path)
        
        
    # @FLOPs_DECORATOR
    def _run(self):
        try:
            self.count = 0.0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            _paths = sorted(glob.glob(f"{self.src_folder_path}/*.png"))
            print(self.src_folder_path)
            paths = random.sample(_paths, min(self.eval_size, len(_paths)))
            logger.info(f"{self.__class__.__name__:<12} : found {len(paths)} files, sampling {self.eval_size}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            for p in paths:
                if self.stop_event.is_set():
                    break
                
                # data_tensor = self.ultra_fast_load(p, device=self.device)
                data_nparray = self.ultra_fast_load(p)
                data_nparray = add_gaussian_noise(data_nparray, sigma=0.1)
                while self.img2od_queue.full():
                    time.sleep(0.01)
                # if __name__ == "__main__":
                #     print(data_nparray.shape)
                self.img2od_queue.put(data_nparray)
                self.count += 1
                
                time.sleep(1/self.fps)  
                
                torch.cuda.empty_cache()
                gc.collect()
                del data_nparray
                
                
            logger.info(f"{self.__class__.__name__:<12} : completed, sending END signal")
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown")

            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            self.energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)

            self.shutdown()

if __name__ == '__main__':
    tmp_p = "../../savedyolo/clean/000001.png"  # path to your PNG
    png_path = Path(tmp_p)
    img = read_image(str(png_path))                
    img_f32 = img.float() / 255                    
    if len(img_f32.shape) == 3:
        image = img_f32[None]
        
    print(f"shape : {tuple(img.shape)}")           # e.g. (3, 720, 1280)
    print(f"min   : {img.min().item()}")           # 0 for uint8 PNGs
    print(f"max   : {img.max().item()}")           # 255 for uint8 PNGs
    print(f"mean  : {img_f32.mean().item():.6f}")  # use float for a precise mean
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_queue = mp.Queue()
    instance = imgStream(tmp_queue, device=device)
    instance.set_config(src_folder_path="../../savedyolo/clean", fps=30, profile_save_path="../../profileyolo", eval_size=10)
    instance.enable_profile(False)
    instance.run()
    instance.shutdown()




