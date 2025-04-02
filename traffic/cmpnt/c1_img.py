import sys
sys.path.append("../")
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
from torch.profiler import profile, record_function, ProfilerActivity


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


class imgStream(Process):
    def __init__(self, img2od_queue, device=None):
        super().__init__(name="ImageStream")
        self.img2od_queue = img2od_queue
        self.stop_event = Event()
        self.device = device
        
    def set_config(self, src_folder_path = None, fps = 30, profile_save_path = None):
        self.src_folder_path = src_folder_path
        self.fps = fps
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"

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
            paths = sorted(glob.glob(f"{self.src_folder_path}/*.pt"))
            logger.info(f"{self.__class__.__name__:<12} : found {len(paths)} files")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            for p in paths:
                if self.stop_event.is_set():
                    break
                
                # data_tensor = self.ultra_fast_load(p, device=self.device)
                data_nparray = self.ultra_fast_load(p, device=None)
                
                self.img2od_queue.put(data_nparray)
                
                time.sleep(1/self.fps)  
                
                # del data_tensor
                del data_nparray
                torch.cuda.empty_cache()
                
            logger.info(f"{self.__class__.__name__:<12} : completed, sending END signal")
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.error(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown")
            self.shutdown()
                
    def shutdown(self):
        self.img2od_queue.put(None)
        self.stop_event.set()  
        
                        
    def ultra_fast_load(self, filepath, device=None):
        with open(filepath, 'rb') as f:
            # Step 1: read number of shape dimensions (1 byte)
            shape_dim = np.frombuffer(f.read(1), dtype=np.int8)[0]

            # Step 2: read the shape (8 bytes per dimension)
            shape = np.frombuffer(f.read(8 * shape_dim), dtype=np.int64)

            # Step 3: read dtype string length (1 byte)
            dtype_len = np.frombuffer(f.read(1), dtype=np.int8)[0]

            # Step 4: read the dtype string
            dtype_str = f.read(dtype_len).decode('ascii')  # e.g. '<f4'

            # Step 5: interpret the rest of the file as data
            np_dtype = np.dtype(dtype_str)
            tensor_data = f.read()

            np_array = np.frombuffer(tensor_data, dtype=np_dtype).copy().reshape(shape)
            
            # return np_array
        
            tensor = torch.from_numpy(np_array)

            # Optional: move to device
            if device is not None:
                tensor = tensor.to(device)
                return tensor
            else:
                return np_array
    
    
if __name__ == "__main__":
    pass
