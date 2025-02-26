"""

                                  ___ face recognition _________
                                 /                              \______ language model 
                                /                               /                     \ 
data ----- object detection ---|----- license plate recognition                        \ 
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
import torch
import numpy as np
import time
import functools
from transformers import GPT2Tokenizer, GPT2Model
import timm
import glob
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from base_app import BaseApp
from tqdm import tqdm
import threading
from queue import Queue
import pdb

class TrafficApp:
    def __init__(self, device=None):
        self.device = device
        self.init_queue()
        
    def init_queue(self):
        self.img2od = Queue()
        self.od2fr = Queue()
        self.od2lpr = Queue()
        self.od2cap = Queue()
        self.fr2lm = Queue()
        self.lpr2lm = Queue()
        self.lm2lm = Queue()
        self.cap2lm = Queue()
        self.running = True

    def img_stream(self, folder_path="", fps=30):
        paths = sorted(glob.glob(f"{folder_path}/*.pt"))
        self.img_cnt = len(paths)
        for p in tqdm(paths, desc="sending data to object detection"):
            if not self.running:
                break
            image_tensor = torch.load(p, weights_only=True, map_location=self.device)
            self.img2od.put(image_tensor)
            time.sleep(1.0 / fps)
        # self.running = False
   
    def od_steam(self, model_id, th=0.25):
        from model_zoo import load_from_pretrained
        od_model, od_proc = load_from_pretrained(model_id, self.device)
        
        while self.running:
            img = self.img2od.get()
            od_output = od_model(img)
            _output = od_proc(od_output,
                           threshold = th,
                           target_size = img.shape[1:])[0]
            scores, labels, boxes = _output["scores"], _output["labels"], _output["boxes"]
            combined = torch.cat((boxes, 
                              scores.unsqueeze(1), 
                              labels), dim=1)
            pdb.set_trace()
            self.od2fr.put(combined)
            self.od2lpr.put(combined)
            self.od2cap.put(combined)

    def fr_stream(self):
        fr_proc = AutoImageProcessor.from_pretrained("microsoft/resnet-152", use_fast=True)
        fr_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152").to(self.device)
        fr_model.eval()
        
        while self.running:
            data = self.od2fr.get()
            # Process data with fr_model
            pass

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_app = TrafficApp(device)
    
    threads = []
    
    img_thread = threading.Thread(target=test_app.img_stream, args=("./sample_data/",))
    od_thread = threading.Thread(target=test_app.od_steam, args=(0, 0.25))
    fr_thread = threading.Thread(target=test_app.fr_stream)
    
    threads = [img_thread, od_thread, fr_thread]
    
    try:
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
    except KeyboardInterrupt:
        test_app.running = False
        for t in threads:
            t.join()
            
    print("All threads finished")