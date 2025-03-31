from multiprocessing import Process, Event
import os
import sys
import glob
import time
import torch
import numpy as np
import logging
from queue import Empty
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sqlite3
sys.path.append("../")
from flops import FLOPs_DECORATOR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
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
        
    def set_config(self, embedding_path):
        self.embedding_path = embedding_path
        
    def shutdown(self):
        self.kr2lm_queue.put(None)
        self.stop_event.set()
        
    @FLOPs_DECORATOR
    def run(self):
        try:
            embed_num = self.load_face_embeddings()
            logger.info(f"{self.__class__.__name__:<12} : face embeddings loaded, found {embed_num} embeddings")
            logger.info(f"{self.__class__.__name__:<12} : started")
            
            fr_end_received = False
            lpr_end_received = False
            
            while not self.stop_event.is_set():
                if fr_end_received and lpr_end_received:
                    break

                if not fr_end_received:
                    try:
                        data = self.fr2kr_queue.get(timeout=2.0)
                        if data is None:  # End signal
                            fr_end_received = True
                            logger.info(f"{self.__class__.__name__:<12} : Received end signal from FR")
                        else:
                            self.process_fr_data(data)
                    except Empty:
                        continue
                    
                if not lpr_end_received:
                    try:
                        data = self.lpr2kr_queue.get(timeout=2.0)
                        if data is None:  # End signal
                            lpr_end_received = True
                            logger.info(f"{self.__class__.__name__:<12} : Received end signal from LPR")
                        else:
                            self.process_lpr_data(data)
                    except Empty:
                        continue
                
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received BOTH END signal, shutting down")
            self.shutdown()
            
            
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
            
        return len(self.stored_embeddings)
            
        
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