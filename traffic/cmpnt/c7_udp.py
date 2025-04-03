from multiprocessing import Process, Event
import sys
sys.path.append("../")
import time
import torch
import logging
from queue import Empty
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
from dotenv import load_dotenv
import os
import multiprocessing as mp
from flops import FLOPs_DECORATOR, write_profile, write_profile_lt
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule
import socket

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

class udpStream(Process):
    def __init__(self, cap2lm_queue, kr2lm_queue, device=None):
        super().__init__(name="UDPStream")
        self.cap2lm_queue = cap2lm_queue
        self.kr2lm_queue = kr2lm_queue
        self.stop_event = Event()
        self.device = device         
        
    def set_config(self, model_id="gpt2", profile_save_path = None):
        self.model_id = model_id
        self.profile_save_path = profile_save_path + f"/{self.__class__.__name__}"
        
    def shutdown(self):
        self.stop_event.set()
        try:
            if hasattr(self, 'model'):
                try:
                    self.model = self.model.to("cpu")
                except:
                    pass
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.tokenizer = None
                
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : error in shutting down {str(e)}")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : shutdown successful")
            
    def run(self):    
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True,
                     profile_memory=False,
                     record_shapes=False
                     ) as prof:
            with record_function(f"{self.__class__.__name__}"):
                self._run()
        write_profile(prof, self.profile_save_path)

    # @FLOPs_DECORATOR
    def _run(self):
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"{self.__class__.__name__:<12} : LM model loaded, model_id = {self.model_id}")
        logger.info(f"{self.__class__.__name__:<12} : started")
        try:            
            cap_end_received = False
            kr_end_received = False
            
            while not self.stop_event.is_set():
                if cap_end_received and kr_end_received:
                    break

                if not cap_end_received:
                    try:
                        data = self.cap2lm_queue.get(block=False)
                        if data is None:
                            cap_end_received = True
                            logger.info(f"{self.__class__.__name__:<12} : Received end signal from CAP")
                        else:
                            # self.gpt2(data, max_new_tokens=50)
                            self.send_message_udp(data)
                    except Empty:
                        time.sleep(0.1)
                        pass
                    
                if not kr_end_received:
                    try:
                        data = self.kr2lm_queue.get(block=False)
                        if data is None:
                            kr_end_received = True
                            logger.info(f"{self.__class__.__name__:<12} : Received end signal from KR")
                        else:
                            # self.gpt2(data, max_new_tokens=50)
                            self.send_message_udp(data)
                    except Empty:
                        time.sleep(0.1)
                        pass

                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"{self.__class__.__name__:<12} : {str(e)}")
        except KeyboardInterrupt:
            logger.info(f"{self.__class__.__name__:<12} : Interrupted by user")
        finally:
            logger.info(f"{self.__class__.__name__:<12} : Received BOTH END signal, shutting down")
            self.shutdown()

            
    def grok2(self, content, model="grok-2-latest", stream=False, temperature=0, max_tokens=50):
        """Make an API call to the Grok-2 API"""
        load_dotenv()
        api_key = os.getenv("GROK_2_API_KEY")
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "messages": [{"role": "user", "content": content}],
            "model": model,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens 
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
                    
                    
    def gpt2(self, prompt, max_new_tokens=50):
        if isinstance(prompt, str):
            prompt = " ".join(prompt)
        else:
            raise ValueError("Prompt must be a string")
        
        with torch.no_grad():
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                **encoded_input, 
                max_new_tokens=max_new_tokens, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return decoded_text
    
    def send_message_udp(self, message):
        host = '127.0.0.1'  
        port = 65432        
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
            udp_socket.sendto(message.encode('utf-8'), (host, port))


if __name__ == "__main__":
    import torch
    import time
    import warnings
    import onnxruntime as ort
    from transformers import AutoTokenizer, pipeline
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    import os
    
    warnings.filterwarnings("ignore", message="Device set to use cpu")
    warnings.filterwarnings("ignore", message="For the decoder with past, using ONNX models")

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Force CPU for PyTorch
    device = torch.device("cpu")
    
    # Correct provider specification
    providers = ["CPUExecutionProvider"]
    
    session_options = ort.SessionOptions()

    tokenizer = AutoTokenizer.from_pretrained("optimum/t5-small")
    model = ORTModelForSeq2SeqLM.from_pretrained(
        "optimum/t5-small", 
        providers=providers,
        # Remove provider_options as it's causing errors
    )
    
    print(f"Model loaded: {model}")
    start_time = time.perf_counter()
    translator = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer, device=-1)
    results = translator("My name is Eustache and I have a pet raccoon")
    end_time = time.perf_counter()
    print(f"Translation time: {end_time - start_time:.4f} seconds")
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Translation result: {results}")