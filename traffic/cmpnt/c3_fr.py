from multiprocessing import Process, Queue, Event
from queue import Empty
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1 # https://github.com/timesler/facenet-pytorch/tree/master
from PIL import Image
import torch
import sys
sys.path.append("../")
from flops import FLOPs_DECORATOR
import torch.nn.functional as F
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class frStream(Process):
    def __init__(self, od2fr_queue, fr2kr_queue, device=None):
        super().__init__(name="FRStream")
        self.od2fr_queue = od2fr_queue
        self.fr2kr_queue = fr2kr_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_config(self, model_id="vggface2"):
        self.model_id = model_id

    @FLOPs_DECORATOR
    def run(self):
        try:
            self.facenet = InceptionResnetV1(pretrained=f"{self.model_id}").eval().to(self.device)
            logger.info(f"{self.__class__.__name__:<12} : FR model loaded, model_id = {self.model_id}")
            logger.info(f"{self.__class__.__name__:<12} : started")
            while not self.stop_event.is_set():
                # Get next image with timeout
                try:
                    data = self.od2fr_queue.get(timeout=2.0)
                    if data is None:  # End signal
                        break
                except Empty:
                    continue
                
                # Process the image
                data_tensor = torch.from_numpy(data).to(self.device)
                padded_image = self.facenet_padding(data_tensor)
                
                with torch.no_grad():
                    face_embedding = self.facenet(padded_image).cpu().numpy()
                    
                self.fr2kr_queue.put(face_embedding)
                
                del data_tensor, padded_image, face_embedding
                torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : interruptted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received END signal, shutting down")
            self.shutdown()
            
    def shutdown(self):
        self.stop_event.set()
        self.fr2kr_queue.put(None)

    def facenet_padding(self, image, min_size=160):
        # Get current dimensions
        if image.dim() == 4:
            image = image.squeeze(0)
            
        _, height, width = image.shape
        
        # Check if padding is needed
        if height < min_size or width < min_size:
            # Calculate padding
            pad_height = max(0, min_size - height)
            pad_width = max(0, min_size - width)
            
            # Calculate padding for each side
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # Apply padding
            image = F.pad(image.unsqueeze(0), 
                        (pad_left, pad_right, pad_top, pad_bottom), 
                        mode='constant', value=0)
            image = image.squeeze(0)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image