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
import pickle

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
                    
                # image_tensor = torch.load(p, weights_only=False, pickle_module=pickle, map_location=self.device)
                # numpy_array = image_tensor.cpu().numpy()

                numpy_array = self.ultra_fast_load(p, device=self.device)
                
                # Send numpy array through queue
                self.img2od_queue.put(numpy_array)
                
                # Simulate frame rate
                time.sleep(1.0 / self.fps)
                
                # Clean up
                del numpy_array
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
        
    def ultra_fast_load(self, filepath, device=None):
        with open(filepath, 'rb') as f:
            # Read shape dimensions
            shape_bytes = f.read(24)  # Assuming at most 3 dimensions
            shape = np.frombuffer(shape_bytes, dtype=np.int64)
            # Filter out any zeros in the shape
            shape = shape[shape > 0]
            
            # Read dtype (1 byte)
            dtype_num = np.frombuffer(f.read(1), dtype=np.int8)[0]
            
            # Map numpy dtype number to torch dtype and corresponding numpy dtype
            dtype_map = {
                1: (torch.float32, np.float32),  # np.float32
                2: (torch.float64, np.float64),  # np.float64
                3: (torch.complex64, np.complex64),  # np.complex64
                4: (torch.complex128, np.complex128),  # np.complex128
                5: (torch.int8, np.int8),  # np.int8
                6: (torch.int16, np.int16),  # np.int16
                7: (torch.int32, np.int32),  # np.int32
                8: (torch.int64, np.int64),  # np.int64
                9: (torch.uint8, np.uint8),  # np.uint8
                # Add more mappings as needed
            }
            
            torch_dtype, numpy_dtype = dtype_map.get(dtype_num, (torch.float32, np.float32))
            
            # Read tensor data
            tensor_data = f.read()
            
            # Convert to numpy array then torch tensor
            np_array = np.frombuffer(tensor_data, dtype=numpy_dtype).reshape(shape)
            tensor = torch.from_numpy(np_array)
            
            # Move to device if specified
            if device is not None:
                tensor = tensor.to(device)
                
            return tensor

    def ultra_fast_load(self, filepath, device=None):
        with open(filepath, 'rb') as f:
            fd = f.fileno()
            try:
                shape_dims = int(np.frombuffer(os.read(fd, 1), dtype=np.int8)[0])
                shape = tuple(np.frombuffer(os.read(fd, shape_dims * 8), dtype=np.int64))
                dtype_length = int(np.frombuffer(os.read(fd, 1), dtype=np.int8)[0])
                dtype_str = os.read(fd, dtype_length).decode('ascii')

                data = os.read(fd, os.path.getsize(fd) - (1 + shape_dims * 8 + 1 + dtype_length))
                tensor = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
                return tensor
                return torch.from_numpy(tensor).clone().to(device)

            except Exception as e:
                print(f"Failed to load tensor from {filepath}: {e}")
                return None

def ultra_fast_load(filepath):
    with open(filepath, 'rb') as f:
        fd = f.fileno()
        try:
            shape_dims = int(np.frombuffer(os.read(fd, 1), dtype=np.int8)[0])
            shape = tuple(np.frombuffer(os.read(fd, shape_dims * 8), dtype=np.int64))
            dtype_length = int(np.frombuffer(os.read(fd, 1), dtype=np.int8)[0])
            dtype_str = os.read(fd, dtype_length).decode('ascii')

            data = os.read(fd, os.path.getsize(fd) - (1 + shape_dims * 8 + 1 + dtype_length))
            tensor = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
            return torch.from_numpy(tensor)
        except Exception as e:
            print(f"Failed to load tensor from {filepath}: {e}")
            return None
            
if __name__ == "__main__":
    model_id = 0
    algorithm = "teastatic"
    target_idx = 68
    data_path = f"../saved/model_{model_id}/{algorithm}_tgt_{str(target_idx).lower()}"
    print(data_path)
    paths = sorted(glob.glob(f"{data_path}/*.pt"))
    
    print(f"Found {len(paths)} image files")
    
    p = paths[0]
    
    image_tensor = ultra_fast_load(p)
