import random
from datasets import load_dataset
import CONSTANTS
import torch
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2
import logging
from PIL import Image, ImageDraw
from torchvision.utils import save_image
random.seed(42)

if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000001000.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:1')

    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    t = inputs["pixel_values"]
    print(t.shape)