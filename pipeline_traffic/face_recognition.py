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
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1 # https://github.com/timesler/facenet-pytorch/tree/master
from PIL import Image
import cv2
import torch.nn.functional as F


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
            # self.mtcnn = MTCNN(image_size=160, 
            #                    margin=20, 
            #                    min_face_size=20,
            #                    thresholds=[0.6, 0.7, 0.7], 
            #                    factor=0.709, 
            #                    post_process=True,
            #                    device=self.device,
            #                    select_largest=True  # Select largest face only
            #                    )
            logger.info("Loading stored embeddings from ./face_embeddings")
            self.stored_embeddings = {}
            embeddings_path = "./face_embeddings"
            for embedding_file in glob.glob(os.path.join(embeddings_path, "*.npy")):
                basename = os.path.basename(embedding_file)
                name = basename.rsplit(".", 1)[0]  # Just remove the .npy extension
                    
                # Load the embedding
                embedding = np.load(embedding_file)
                self.stored_embeddings[name] = embedding
                
            logger.info(f"Loaded {len(self.stored_embeddings)} embeddings")
            # self.facenet.classify = True
            
            logger.info("FR Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2fr_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # legacy code
                    # if isinstance(data, str) and data == "END OF FRAME":
                    #     self.fr2kr_queue.put("END OF FRAME")
                    #     continue
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    request = self.facenet_padding(request)
                    # request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        # to_pil = transforms.ToPILImage()
                        # pil_image = to_pil(request[0])
                        embeddings = self.facenet(request)
                        current_embedding = embeddings.cpu().numpy()
                        
                        # Find the most similar face
                        best_match, similarity = self.find_most_similar(current_embedding)
                    
                        result_string = f"{best_match}|{float(similarity):.4f}"

                        # predicted_label = embeddings[0].argmax()
                    
                    self.fr2kr_queue.put(str(result_string))
                    
                    del request, embeddings, current_embedding, best_match, similarity, result_string
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
        
        
    def find_most_similar(self, current_embedding):
        """
        Find the most similar face embedding from the stored embeddings.
        
        Args:
            current_embedding: numpy array of the current face embedding
            
        Returns:
            tuple: (name of the most similar face, similarity score)
        """
        best_match = None
        best_similarity = -1.0
        
        # Flatten the current embedding if needed
        if current_embedding.ndim > 1:
            current_embedding = current_embedding.flatten()
        
        # Normalize the current embedding
        current_embedding_norm = current_embedding / np.linalg.norm(current_embedding)
        
        # Compare with all stored embeddings
        for name, stored_embedding in self.stored_embeddings.items():
            # Flatten the stored embedding if needed
            if stored_embedding.ndim > 1:
                stored_embedding = stored_embedding.flatten()
            
            # Normalize the stored embedding
            stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding_norm, stored_embedding_norm)
            
            # Update best match if this is more similar
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return best_match, best_similarity
    
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