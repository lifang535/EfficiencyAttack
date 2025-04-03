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
import torch.nn.functional as F
import sqlite3

    # Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class krStream(Process):
    def __init__(self, fr2kr_queue, lpr2kr_queue, kr2lm_queue, device=None):
        super().__init__(name="KRStream")
        self.fr2kr_queue = fr2kr_queue
        self.lpr2kr_queue = lpr2kr_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = None
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    @calculate_flops_decorator
    def run(self):
        try:
            logger.info("KR stream started")
            
            logger.info(f"Loading KR model {self.model_id}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("KR Model loaded and ready")
            
            # Track end signals from upstream processes
            fr_end_received = False
            lpr_end_received = False
            
            # Load face embeddings
            self.load_face_embeddings()
                        
            while not self.stop_event.is_set():
                try:
                    # Exit if both upstream processes have ended
                    if fr_end_received and lpr_end_received:
                        logger.info("Both FR and LPR streams have ended, terminating KR stream")
                        break
                    
                    # Process face recognition data if available
                    if not fr_end_received and not self.fr2kr_queue.empty():
                        data = self.fr2kr_queue.get(block=False)0)
                        if data is None:
                            fr_end_received = True
                            logger.info("Received end signal from FR stream")
                        else:
                            self.process_fr_data(data)
                    
                    # Process license plate data if available
                    elif not lpr_end_received and not self.lpr2kr_queue.empty():
                        data = self.lpr2kr_queue.get(block=False)0)
                        if data is None:
                            lpr_end_received = True
                            logger.info("Received end signal from LPR stream")
                        else:
                            self.process_lpr_data(data)
                    
                    # If both queues are empty but processes haven't ended, wait briefly
                    elif self.fr2kr_queue.empty() and self.lpr2kr_queue.empty():
                        time.sleep(0.1)
                        continue
                    
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing data streams: {str(e)}", exc_info=True)
                    
            # Signal end to LM stream
            logger.info("KR stream sending end signal to LM stream")
            self.kr2lm_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in KR stream: {str(e)}", exc_info=True)
            self.kr2lm_queue.put(None)
        finally:
            logger.info("KR stream ended")
            self.kr2lm_queue.put(None)
            self.shutdown()
            
            
    def shutdown(self):
        """Signal the process to stop"""
        self.stop_event.set()
    
    def load_face_embeddings(self):
        """Load face embeddings from disk"""
        self.stored_embeddings = {}
        embeddings_path = "./face_embeddings"
        
        for embedding_file in glob.glob(os.path.join(embeddings_path, "*.npy")):
            basename = os.path.basename(embedding_file)
            name = basename.rsplit(".", 1)[0]  # Remove the .npy extension
                
            # Load the embedding
            embedding = np.load(embedding_file)
            self.stored_embeddings[name] = embedding
            
        logger.info(f"Loaded {len(self.stored_embeddings)} face embeddings")
        
    def process_fr_data(self, data):
        """Process face recognition data"""
        with torch.no_grad():
            best_match, similarity = self.find_most_similar(data)
            result_string = f"{best_match}|{float(similarity):.4f}"
            self.kr2lm_queue.put(result_string)
            
            # Clean up
            del data, best_match, similarity, result_string
            torch.cuda.empty_cache()
    
    def process_lpr_data(self, data):
        """Process license plate recognition data"""
        with torch.no_grad():
            query_result = self.db_query(data)
            self.kr2lm_queue.put(query_result)
            
            # Clean up
            del query_result
            torch.cuda.empty_cache()
        
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
    
    def db_query(self, plate):
        conn = sqlite3.connect("us_license_plates.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT plate, state FROM license_plates WHERE plate = ?", (plate,))
        result = cursor.fetchone()
        
        if result:
            conn.close()
            return f"Plate found: {result[0]}, state: {result[1]}"
        else:
            return "Plate not found"
            cursor.execute("INSERT INTO license_plates (plate, state) VALUES (?, ?)", (plate, "TX"))
            conn.commit()
            conn.close()
            return "Plate not found, inserted into database"
        