import sys
import os
folder_path = "../vit/"
sys.path.append(folder_path)
from ms_captioning import MSCaptioning
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

def detr():
  parser = argparse.ArgumentParser(description="DETR hyperparam setup")
  parser.add_argument("--e", type=int, default=-999)
  parser.add_argument("--t", type=str, default="infer")
  args = parser.parse_args()
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  print("running on : ", device)
  image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
  model.eval()
  to_tensor = transforms.ToTensor()
  results_dict = {}
  coco_data = load_dataset("detection-datasets/coco", split="val")
