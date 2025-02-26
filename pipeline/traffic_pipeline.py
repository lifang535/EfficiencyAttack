from multiprocessing import Process, Queue, Value
from queue import Empty
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import glob
import time
import torch
import sys
sys.path.append("../")
from model_zoo import load_from_pretrained
from tqdm import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


class imgStream(Process):
    
    def __init__(self, img2od_queue: Queue, device=None):
        super().__init__()
        self.img2od_queue = img2od_queue
        self.end_flag = False
        self.device = device
        
        
    def set_config(self, folder_path="", fps=30):
        self.folder_path = folder_path
        self.fps = fps
        
        
    def run(self):
        self.read_img()
        self._end()
        
        
    def read_img(self):
        paths = sorted(glob.glob(f"{self.folder_path}/*.pt"))
        self.img_cnt = len(paths)
        for p in tqdm(paths, desc="sending data to object detection"):
            image_tensor = torch.load(p, weights_only=True, map_location=self.device)
            self.img2od_queue.put(image_tensor.clone())
            time.sleep(1.0 / self.fps)
            
            del image_tensor
            torch.cuda.empty_cache()

   
    def _end(self):
        self.img2od_queue.put(None)
        self.end_flag = True
        print("imgStream ended")
        
        
class odStream(Process):
    def __init__(self, 
                 img2od_queue: Queue, 
                 od2fr_queue: Queue,
                 od2lpr_queue: Queue,
                 od2cap_queue: Queue,
                 device=None):
        super().__init__()
        self.img2od_queue = img2od_queue
        self.od2fr_queue = od2fr_queue
        self.od2lpr_queue = od2lpr_queue
        self.od2cap_queue = od2cap_queue
        self.end_flag = False
        self.device = device
        
        
        
    def set_config(self, model_id):
        self.od_model, self.od_processor = load_from_pretrained(model_id)
        self.od_model.to(self.device)
    
    
    def run(self):
        self.object_detection()
        
        
    def object_detection(self):
        while not self.end_flag:
            try:
                request = self.img2od_queue.get(timeout=1)

                
            except Empty:
                continue
            if request is None:
                self._end()
                break
            
            request = request.unsqueeze(0).to(self.device)
            
            od_output = self.od_model(request)
            _output = self.od_processor.post_process_object_detection(od_output,
                                                                      threshold = 0.25,
                                                                      target_sizes = [request.shape[2:] for _ in range(1)])[0]
            scores, labels, boxes = _output["scores"], _output["labels"], _output["boxes"]
            combined = torch.cat((boxes, 
                                  scores.unsqueeze(1), 
                                  labels.unsqueeze(1)), dim=1)
            
            del request
            del od_output
            del _output
            del scores, labels, boxes
            del combined
            torch.cuda.empty_cache()
    
    
    def _end(self):
        self.od2fr_queue.put(None)
        self.od2lpr_queue.put(None)
        self.od2cap_queue.put(None)
        self.end_flag = True
        print("odStream ended")
     
     
   
def clean_up(process_list, queue_list):
    for q in queue_list:
        while not q.empty():
            q.get_nowait()
        del q
    for process in process_list:
        if process.is_alive():
            process.terminate()
            process.close()
            process.join()

    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn', force=True)

    img2od = Queue()
    od2fr = Queue()
    od2lpr = Queue()
    od2cap = Queue()
    fr2lm = Queue()
    lpr2lm = Queue()
    lm2lm = Queue()
    cap2lm = Queue()
    
    

    
    end_signal = Value('b', False)
    
    img_stream = imgStream(img2od, device)
    od_stream = odStream(img2od, od2fr, od2lpr, od2cap, device)
    
    img_stream.set_config(folder_path="./sample_data")
    od_stream.set_config(model_id=0)
    
    img_stream.start()
    od_stream.start()

    queues = [img2od, od2fr, od2lpr, od2cap, fr2lm, lpr2lm, lm2lm, cap2lm]
    processes = [img_stream, od_stream]
    
    try:
        torch.cuda.empty_cache()
        img_stream.join()
        od_stream.join()
        
        end_signal.value = True
        
    except KeyboardInterrupt:
        clean_up(processes, queues)
        img_stream.terminate()
        od_stream.terminate()
        
    finally: 
        clean_up(processes, queues)

        img_stream.close()
        od_stream.close()
        torch.cuda.empty_cache()