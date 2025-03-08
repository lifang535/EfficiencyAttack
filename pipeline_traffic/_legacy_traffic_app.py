"""

                                  ___ face recognition _________
                                 /                              \______ knowledge ___
                                /                               /       retrieval    \ 
data ----- object detection ---|----- license plate recognition                       \     
                                \                                                      |--- language model
                                 \                                                    /
                                  \___ image captioning _____________________________/


object detection:
    - YOLO or Vision Transformer
    - Input: PIL Image
    - Output: (box, cls, scores)
    
face recognition:
    - ResNet 101
    - Input: bounding boxes which has a cls label == "person"
    - Output: 
    
license plate recognition:
    - ResNet 18
    - Input: bounding boxes which has a cls label == "car"
    - Output:
    
image captioning:
    - ResNet 101
    - Input: bounding boxes which has a cls label == ["person", "car", "traffic lights", "stop sign"]
    - Output:
        
language model:
    - GPT 2
    
"""


from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
import os
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model_zoo import load_from_pretrained

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append("../")


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
                    
                # Load image to device memory
                image_tensor = torch.load(p, weights_only=True, map_location=self.device)
                
                # Convert to numpy array (safer for multiprocessing)
                numpy_array = image_tensor.cpu().numpy()
                
                # Send numpy array through queue
                self.img2od_queue.put(numpy_array)
                
                # Simulate frame rate
                time.sleep(1.0 / self.fps)
                
                # Clean up
                del image_tensor, numpy_array
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


class odStream(Process):
    def __init__(self, img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device=None):
        super().__init__(name="ODStream")
        self.img2od_queue = img2od_queue
        self.od2fr_queue = od2fr_queue
        self.od2lpr_queue = od2lpr_queue
        self.od2cap_queue = od2cap_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
    
    def run(self):
        try:
            logger.info("OD stream started")
            
            # Load model
            logger.info(f"Loading OD model {self.model_id}")
            self.model, self.processor = load_from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("OD Model loaded and ready")
            
            # Process images
            while not self.stop_event.is_set():
                try:
                    # Get next image with timeout
                    data = self.img2od_queue.get(timeout=2.0)
                    
                    # Check for end signal
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        od_output = self.model(request)
                        
                    # Process detections
                    _output = self.processor.post_process_object_detection(
                        od_output,
                        threshold=0.25,
                        target_sizes=[request.shape[2:]]
                    )[0]
                    
                    # Extract results
                    scores, labels, boxes = _output["scores"], _output["labels"], _output["boxes"]
                    
                    """
                    ===============================================================================
                    from all bounding boxes, select ONLY "person" boxes for face recognition
                    """
                    found_valid_box = False
                    for label, box in zip(labels, boxes):
                        if int(label.item()) == 0:
                            found_valid_box = True
                            x1, y1, x2, y2 = box
                            cropped_tensor = request[:, :, int(y1):int(y2), int(x1):int(x2)]
                            cropped_np = cropped_tensor.cpu().numpy()
                            self.od2fr_queue.put(cropped_np)
                            
                    if found_valid_box:
                        pass
                    else:
                        self.od2fr_queue.put(request.cpu().numpy())
                        
                    self.od2fr_queue.put("END OF FRAME")
                    """
                    ===============================================================================
                    from all bounding boxes, select ONLY "car" boxes for license plate recognition
                    """
                    found_valid_box = False
                    for label, box in zip(labels, boxes):
                        if int(label.item()) == 1:
                            found_valid_box = True
                            x1, y1, x2, y2 = box
                            cropped_tensor = request[:, :, int(y1):int(y2), int(x1):int(x2)]
                            cropped_np = cropped_tensor.cpu().numpy()
                            self.od2lpr_queue.put(cropped_np)   

                    if found_valid_box:
                        pass
                    else:
                        self.od2lpr_queue.put(request.cpu().numpy())
                        
                    self.od2lpr_queue.put("END OF FRAME")
                    
                    """
                    ===============================================================================
                    from all bounding boxes, select traffic related boxes for image captioning
                    ["person", "car", "traffic lights", "stop sign", etc...]
                    """
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = float('-inf'), float('-inf')
                    found_valid_box = False
                    for label, box in zip(labels, boxes):
                        x1, y1, x2, y2 = box
                        if int(label.item()) in [0, 5, 1, 11, 2, 12, 9, 6, 7, 3]:
                            min_x = min(min_x, x1.item())
                            min_y = min(min_y, y1.item())
                            max_x = max(max_x, x2.item())
                            max_y = max(max_y, y2.item())
                            found_valid_box = True
                    if found_valid_box:
                        cropped_tensor = request[:, :, int(min_y):int(max_y), int(min_x):int(max_x)]
                        cropped_np = cropped_tensor.cpu().numpy()
                        self.od2cap_queue.put(cropped_np)
                    else:
                        self.od2cap_queue.put(request.cpu().numpy())
                        
                    # Clean up
                    del request, od_output, _output, scores, labels, boxes
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("img2od Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing img2od Queue: {str(e)}", exc_info=True)
            
            # Signal end to downstream processes
            for q in [self.od2fr_queue, self.od2lpr_queue, self.od2cap_queue]:
                q.put(None)
                
        except Exception as e:
            logger.error(f"Error in OD stream: {str(e)}", exc_info=True)
            # Signal end to downstream processes on error
            for q in [self.od2fr_queue, self.od2lpr_queue, self.od2cap_queue]:
                q.put(None)
        finally:
            logger.info("OD stream ended")
    
    def shutdown(self):
        self.stop_event.set()


class frStream(Process):
    def __init__(self, od2fr_queue, fr2kr_queue, device=None):
        super().__init__(name="FRStream")
        self.od2fr_queue = od2fr_queue
        self.fr2kr_queue = fr2kr_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    def run(self):
        try:
            logger.info("FR stream started")
            
            logger.info(f"Loading FR model {self.model_id}")
            self.model = ResNetForImageClassification.from_pretrained(self.model_id)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FR Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2fr_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    if isinstance(data, str) and data == "END OF FRAME":
                        self.fr2kr_queue.put("END OF FRAME")
                        continue
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    # request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        logits = self.model(request).logits
                        predicted_label = logits.argmax(-1).item()
                    
                    self.fr2kr_queue.put(str(predicted_label))
                    
                    del request, logits, predicted_label
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


class lprStream(Process):
    def __init__(self, od2lpr_queue, lpr2kr_queue, device=None):
        super().__init__(name="LPRStream")
        self.od2lpr_queue = od2lpr_queue
        self.lpr2kr_queue = lpr2kr_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    def run(self):
        try:
            logger.info("LPR stream started")
            
            logger.info(f"Loading LPR model {self.model_id}")
            self.model = ResNetForImageClassification.from_pretrained(self.model_id)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("LPR Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2lpr_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    if isinstance(data, str) and data == "END OF FRAME":
                        self.lpr2kr_queue.put("END OF FRAME")
                        continue
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    # request = request.unsqueeze(0)  # Add batch dimension
                    
                    with torch.no_grad():
                        logits = self.model(request).logits
                        predicted_label = logits.argmax(-1).item()
                        
                    self.lpr2kr_queue.put(str(predicted_label))
                    
                    del request, logits, predicted_label
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("od2lpr Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2lpr Queue: {str(e)}", exc_info=True)
                    
            self.lpr2kr_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in LPR stream: {str(e)}", exc_info=True)
            self.lpr2kr_queue.put(None)
        finally:
            logger.info("LPR stream ended")
            
    def shutdown(self):
        self.stop_event.set()  


class capStream(Process):
    def __init__(self, od2cap_queue, cap2lm_queue, device=None):
        super().__init__(name="CAPStream")
        self.od2cap_queue = od2cap_queue
        self.cap2lm_queue = cap2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def set_config(self, model_id):
        self.model_id = model_id
        
    def run(self):
        try:
            logger.info("CAP stream started")
            
            logger.info(f"Loading CAP model {self.model_id}")
            self.model = ResNetForImageClassification.from_pretrained(self.model_id)
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("CAP Model loaded and ready")
            
            while not self.stop_event.is_set():
                try:
                    data = self.od2cap_queue.get(timeout=2.0)
                    
                    if data is None:
                        logger.info("Received end signal")
                        break
                    
                    # Convert numpy array back to tensor and move to GPU
                    request = torch.from_numpy(data).to(self.device)
                    # request = request.unsqueeze(0)  
                    
                    with torch.no_grad():
                        logits = self.model(request).logits
                        caption = logits.argmax(-1).item()
                        
                    self.cap2lm_queue.put(str(caption))
                    
                    del request, logits, caption
                    torch.cuda.empty_cache()
                    
                except Empty:
                    logger.debug("od2cap Queue timeout, checking if should continue")
                    continue
                except Exception as e:
                    logger.error(f"Error processing od2cap Queue: {str(e)}", exc_info=True)
                    
            self.cap2lm_queue.put(None)
            
        except Exception as e:
            logger.error(f"Error in CAP stream: {str(e)}", exc_info=True)
            self.cap2lm_queue.put(None)
        finally:
            logger.info("CAP stream ended")
            
    def shutdown(self):
        self.stop_event.set()


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
        
    def run(self):
        try:
            logger.info("KR stream started")
            
            logger.info(f"Loading KR model {self.model_id}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("KR Model loaded and ready")
            
            fr_data_list = []
            lpr_data_list = []
            fr_end_received = False
            lpr_end_received = False
            
            while not self.stop_event.is_set():
                try:
                    # Check if both upstream processes have terminated
                    if fr_end_received and lpr_end_received:
                        logger.info("Both FR and LPR streams have ended, terminating KR stream")
                        break
                        
                    # Process FR data
                    try:
                        fr_data = self.fr2kr_queue.get(timeout=1.0)
                        if fr_data is None:
                            fr_end_received = True
                            logger.info("Received end signal from FR stream")
                        elif isinstance(fr_data, str) and fr_data == "END OF FRAME":
                            pass  # Process frame boundary
                        else:
                            fr_data_list.append(fr_data)
                    except Empty:
                        pass
                        
                    # Process LPR data
                    try:
                        lpr_data = self.lpr2kr_queue.get(timeout=1.0)
                        if lpr_data is None:
                            lpr_end_received = True
                            logger.info("Received end signal from LPR stream")
                        elif isinstance(lpr_data, str) and lpr_data == "END OF FRAME":
                            pass  # Process frame boundary
                        else:
                            lpr_data_list.append(lpr_data)
                    except Empty:
                        pass
                        
                    # Process accumulated data if we have both LPR and FR data
                    if fr_data_list and lpr_data_list:
                        prompt = "".join(fr_data_list + lpr_data_list)
                        
                        with torch.no_grad():
                            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                            generated_ids = self.model.generate(**encoded_input, 
                                                                max_new_tokens=50, 
                                                                do_sample=True,
                                                                pad_token_id=self.tokenizer.eos_token_id)
                            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        
                        self.kr2lm_queue.put(decoded_text)
                        
                        del prompt, encoded_input, generated_ids, decoded_text
                        fr_data_list = []
                        lpr_data_list = []
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Error in KR stream processing: {str(e)}", exc_info=True)
            
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


class lmStream(Process):
    def __init__(self, cap2lm_queue, kr2lm_queue, device=None):
        super().__init__(name="LMStream")
        self.cap2lm_queue = cap2lm_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def set_config(self, model_id):
        self.model_id = model_id
        
    def run(self):
        try:
            logger.info("LM stream started")
            
            logger.info(f"Loading LM model {self.model_id}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("LM Model loaded and ready")
            
            cap_end_received = False
            kr_end_received = False
            cap_data = None
            kr_data = None
            while not self.stop_event.is_set():
                # Exit condition: both upstream processes have terminated
                if cap_end_received and kr_end_received:
                    logger.info("Both CAP and KR streams have ended, terminating LM stream")
                    break
                
                try:
                    # Get data from CAP stream with short timeout
                    try:
                        cap_data = self.cap2lm_queue.get(timeout=0.5)
                        if cap_data is None:
                            cap_end_received = True
                            logger.info("Received end signal from CAP stream")
                        else:
                            # Process CAP data
                            pass
                    except Empty:
                        pass
                    
                    # Get data from KR stream with short timeout
                    try:
                        kr_data = self.kr2lm_queue.get(timeout=0.5)
                        if kr_data is None:
                            kr_end_received = True
                            logger.info("Received end signal from KR stream")
                        else:
                            # Process KR data
                            pass
                    except Empty:
                        pass
                    
                    # If we have both cap_data and kr_data, process them
                    if cap_data is not None and kr_data is not None and not isinstance(cap_data, bool) and not isinstance(kr_data, bool):
                        prompt = kr_data + cap_data
                        
                        with torch.no_grad():
                            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                            generated_ids = self.model.generate(**encoded_input, 
                                                                max_new_tokens=50, 
                                                                do_sample=True,
                                                                pad_token_id=self.tokenizer.eos_token_id)                            
                            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            logger.info(f"Generated output: {decoded_text[:50]}...")
                        
                        del prompt, encoded_input, generated_ids, decoded_text
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Error processing in LM stream: {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error in LM stream: {str(e)}", exc_info=True)
        finally:
            logger.info("LM stream ended")
        
    def shutdown(self):
        self.stop_event.set()
        
def clean_up(processes, queues):
    """Clean up processes and queues"""
    logger.info("Cleaning up resources")
    
    # Signal processes to stop
    for p in processes:
        if hasattr(p, 'shutdown') and p.is_alive():
            logger.info(f"Shutting down {p.name}")
            p.shutdown()
    
    # Wait a moment for graceful shutdown
    time.sleep(1.0)
    
    # Force terminate if needed
    for p in processes:
        if p.is_alive():
            logger.info(f"Terminating {p.name}")
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                logger.warning(f"Process {p.name} did not terminate, killing")
                p.kill()
    
    # Drain queues
    for q in queues:
        try:
            while not q.empty():
                _ = q.get_nowait()
        except:
            pass
    
    # Final cleanup
    torch.cuda.empty_cache()
    logger.info("Cleanup complete")

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True,  # Enable FLOP counting
                 profile_memory=True,  # Track memory usage
                 record_shapes=True   # Record tensor shapes
                 ) as prof:       
         
        with record_function("model_inference"):
            """
            FLOPs counting:
            """
            
            
            # Setup
            torch.cuda.empty_cache()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Use spawn for CUDA compatibility
            mp.set_start_method('spawn', force=True)
            
            # Create queues
            img2od = Queue()
            od2fr = Queue()
            od2lpr = Queue()
            od2cap = Queue()
            fr2kr = Queue()
            lpr2kr = Queue()
            kr2lm = Queue()
            cap2lm = Queue()
            
            # Initialize processes
            img_stream = imgStream(img2od, device)
            od_stream = odStream(img2od, od2fr, od2lpr, od2cap, device)
            fr_stream = frStream(od2fr, fr2kr, device)
            lpr_stream = lprStream(od2lpr, lpr2kr, device)
            cap_stream = capStream(od2cap, cap2lm, device)
            kr_stream = krStream(fr2kr, lpr2kr, kr2lm, device)
            lm_stream = lmStream(cap2lm, kr2lm, device)
            
            # Configure processes
            img_stream.set_config(folder_path="./sample_data", fps=30)
            od_stream.set_config(model_id=0)
            fr_stream.set_config(model_id="microsoft/resnet-101")
            lpr_stream.set_config(model_id="microsoft/resnet-18")
            cap_stream.set_config(model_id="microsoft/resnet-101")
            kr_stream.set_config(model_id="gpt2")
            lm_stream.set_config(model_id="gpt2")
            
            # Start processes
            logger.info("Starting processes")
            od_stream.start()  # Start OD process first
            time.sleep(0.5)    # Small delay
            img_stream.start() # Then start image stream
            time.sleep(0.5)    # Small delay
            fr_stream.start()
            lpr_stream.start()
            cap_stream.start()
            time.sleep(0.5)
            kr_stream.start()
            time.sleep(0.5)
            lm_stream.start()
            
            processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, lmStream]
            queues = [img2od, od2fr, od2lpr, od2cap, fr2kr, lpr2kr, kr2lm, cap2lm]
            
            try:
                logger.info("Pipeline running")
                
                # Wait for processes to complete
                logger.info("Waiting for image stream to complete")
                img_stream.join()
                logger.info("Image stream completed")
                
                # Give OD stream time to process any remaining items
                logger.info("Waiting for OD stream to complete")
                od_stream.join(timeout=10.0)
                if od_stream.is_alive():
                    logger.warning("OD stream didn't complete in time, shutting down")
                else:
                    logger.info("OD stream completed")
                    
                fr_stream.join()
                lpr_stream.join()
                cap_stream.join()
                kr_stream.join()
                lm_stream.join()
                
                logger.info("All processes completed")
                
            except KeyboardInterrupt:
                logger.info("Interrupted, shutting down")
            finally:
                clean_up(processes, queues)
                logger.info("Pipeline shutdown complete")
                
    print(prof.key_averages().table(sort_by="cuda_time_total"))

    for evt in prof.key_averages():
        if evt.key == "model_inference":
            print(f"FLOPs: {evt.flops}")