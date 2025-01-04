from transformers import AutoImageProcessor, DetrModel
from PIL import Image
import requests
from transformers import AutoImageProcessor
from transformers import VitDetConfig, VitDetModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import RTDetrConfig, RTDetrModel
import torchvision.transforms as transforms
import torch
from datasets import load_dataset
import dataset
import CONSTANTS
import util
from PIL import Image
import requests
import numpy as np
import json
import pdb
from tqdm import tqdm
import argparse
import overload_attack
import single_attack
import phantom_attack
import stra_attack
import adaptive_attack
from datetime import datetime
import os
import random

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://farm2.staticflickr.com/1149/837387952_4f322eeacb_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("running on : ", device)

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
if image.mode != "RGB":
  image = image.convert("RGB")
    
inputs = image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
  outputs = model(**inputs)
model.eval()
to_tensor = transforms.ToTensor()


# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")
target_size = torch.tensor([image.size[::-1]])
# forward pass
outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, 
                                                        threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                        target_sizes = target_size)[0]
results = util.move_to_cpu(results)
pred_scores, pred_labels, pred_boxes = util.parse_prediction(results)
# the last hidden states are the final query embeddings of the Transformer decoder
# these are of shape (batch_size, num_queries, hidden_size)
print(results)
last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)