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
            
            # legacy code
            # fr_data_list = []
            # lpr_data_list = []
            fr_end_received = False
            lpr_end_received = False
            
            self.stored_embeddings = {}
            embeddings_path = "./face_embeddings"
            for embedding_file in glob.glob(os.path.join(embeddings_path, "*.npy")):
                basename = os.path.basename(embedding_file)
                name = basename.rsplit(".", 1)[0]  # Just remove the .npy extension
                    
                # Load the embedding
                embedding = np.load(embedding_file)
                self.stored_embeddings[name] = embedding
                
            logger.info(f"Loaded {len(self.stored_embeddings)} embeddings")
            
            while not self.stop_event.is_set():
                # try:
                    # # Check if both upstream processes have terminated
                    # if fr_end_received and lpr_end_received:
                    #     logger.info("Both FR and LPR streams have ended, terminating KR stream")
                    #     break
                        
                    # # Process FR data
                    # try:
                    #     fr_data = self.fr2kr_queue.get(timeout=1.0)
                    #     if fr_data is None:
                    #         fr_end_received = True
                    #         logger.info("Received end signal from FR stream")
                    #     # legacy code
                    #     # elif isinstance(fr_data, str) and fr_data == "END OF FRAME":
                    #     #     pass  # Process frame boundary
                    #     else:
                    #         fr_data_list.append(fr_data)
                    # except Empty:
                    #     pass
                        
                    # # Process LPR data
                    # try:
                    #     lpr_data = self.lpr2kr_queue.get(timeout=1.0)
                    #     if lpr_data is None:
                    #         lpr_end_received = True
                    #         logger.info("Received end signal from LPR stream")
                    #     # legacy code
                    #     # elif isinstance(lpr_data, str) and lpr_data == "END OF FRAME":
                    #     #     pass  # Process frame boundary
                    #     else:
                    #         lpr_data_list.append(lpr_data)
                    # except Empty:
                    #     pass
                        
     
                #     # Process accumulated data if we have both LPR and FR data
                #     if fr_data_list and lpr_data_list:
                #         prompt = "".join(fr_data_list + lpr_data_list)
                        
                #         with torch.no_grad():
                #             encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                #             generated_ids = self.model.generate(**encoded_input, 
                #                                                 max_new_tokens=50, 
                #                                                 do_sample=True,
                #                                                 pad_token_id=self.tokenizer.eos_token_id)
                #             decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        
                #         self.kr2lm_queue.put(decoded_text)
                        
                #         del prompt, encoded_input, generated_ids, decoded_text
                #         fr_data_list = []
                #         lpr_data_list = []
                #         torch.cuda.empty_cache()
                    
                # except Exception as e:
                #     logger.error(f"Error in KR stream processing: {str(e)}", exc_info=True)
            
            
                try:
                    if self.fr2kr_queue.empty() and self.lpr2kr_queue.empty():
                        continue
                    
                    elif fr_end_received and lpr_end_received:
                        break
                    
                    elif not fr_end_received and not self.fr2kr_queue.empty():
                        data = self.fr2kr_queue.get(timeout=2.0)
                        if data is None:
                            fr_end_received = True
                            logger.info("Received end signal from FR stream")
                        else:
                            with torch.no_grad():
                                best_match, similarity = self.find_most_similar(data)
                                result_string = f"{best_match}|{float(similarity):.4f}"
                                self.kr2lm_queue.put(result_string)
                                
                                del data, best_match, similarity, result_string
                                pass
                        
                    elif not lpr_end_received and not self.lpr2kr_queue.empty():
                        data = self.lpr2kr_queue.get(timeout=2.0)
                        if data is None:
                            lpr_end_received = True
                            logger.info("Received end signal from LPR stream")
                        else:
                            with torch.no_grad():
                                
                                # legacy code
                                # prompt = f"The OCR system read a vehicle license plate as {data}, \
                                #     but OCR errors might have occurred due to character confusion \
                                #     (such as \"8\" ↔ \"B\", \"2\" ↔ \"Z\", \"5\" ↔ \"S\"). List 5 \
                                #     alternative license plate numbers that closely resemble \
                                #     {data} and could correct possible OCR errors. Also print the \
                                #     SQL query that you would use to retrieve the license plate \
                                #     number from the database."
                                # encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                                # generated_ids = self.model.generate(**encoded_input, 
                                #                                     max_new_tokens=50, 
                                #                                     do_sample=True,
                                #                                     pad_token_id=self.tokenizer.eos_token_id)
                                # decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                                # self.kr2lm_queue.put(decoded_text)
                                # del prompt, encoded_input, generated_ids, decoded_text
                                
                                
                                query_result = self.db_query(data)
                                self.kr2lm_queue.put(query_result)
                                del query_result
                                
                                torch.cuda.empty_cache()
                        
                    else:
                        logger.info("Both FR and LPR streams have ended, terminating KR stream")
                    
                    
                except Empty:
                    logger.debug("FR and LPR streams are idle, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing FR and LPR streams: {str(e)}", exc_info=True)
                    
                    
            # Signal end to LM stream
            logger.info("KR stream sending end signal to LM stream")
            self.kr2lm_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in KR stream: {str(e)}", exc_info=True)
            self.kr2lm_queue.put(None)
        finally:
            logger.info("KR stream ended")
            
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
        