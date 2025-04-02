from image_streaming import imgStream
from object_detection import odStream
from face_recognition import frStream
from license_plate_recognition import lprStream
from image_captioning import capStream
from knowledge_retrieval import krStream
from language_model import lmStream
from pipeline_utils import clean_up
from queue_watch import QueueWatch

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
from dotenv import load_dotenv, set_key, dotenv_values
import argparse

parser = argparse.ArgumentParser(description="Traffic pipeline")
parser.add_argument("--model_id", type=int, required=True)
parser.add_argument("--algorithm", type=str, required=True)
parser.add_argument("--target_idx", type=int, default=None)
args = parser.parse_args()

data_path = f"../saved/model_{args.model_id}/{args.algorithm}_tgt_{str(args.target_idx).lower()}"

dotenv_path = ".env"  
load_dotenv(dotenv_path)
set_key(dotenv_path, "SESSION_ID", f"model_{args.model_id}/{args.algorithm}_tgt_{str(args.target_idx)}")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print("\n"*2 + "="*80)
    print("|" + " "*78 + "|")
    print("|" + " "*78 + "|")
    print("|" + " "*21 + "Running: Traffic Monitoring Pipeline" + " "*21 + "|")
    print("|" + " "*78 + "|")
    print("|" + " "*78 + "|")
    print("|" + " "*30 + date_time + " "*29 + "|")
    print("|" + " "*78 + "|")
    print("|" + " "*78 + "|")
    print("="*80 + "\n"*2)
    
    
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method("spawn")
    
    img2od = Queue()
    od2fr = Queue()
    od2lpr = Queue()
    od2cap = Queue()
    fr2kr = Queue()
    lpr2kr = Queue()
    kr2lm = Queue()
    cap2lm = Queue()
    
    img_stream = imgStream(img2od, device)
    od_stream = odStream(img2od, od2fr, od2lpr, od2cap, device)
    fr_stream = frStream(od2fr, fr2kr, device)
    lpr_stream = lprStream(od2lpr, lpr2kr, device)
    cap_stream = capStream(od2cap, cap2lm, device)
    kr_stream = krStream(fr2kr, lpr2kr, kr2lm, device)
    lm_stream = lmStream(cap2lm, kr2lm, device)
    
    queue_watch = QueueWatch({"object detection": img2od, 
                              "face recognition": od2fr, 
                              "license plate recognition": od2lpr, 
                              "captioning": od2cap, 
                              "knowledge retrieval 1": fr2kr, 
                              "knowledge retrieval 2": lpr2kr,
                              "language model 1": cap2lm,
                              "language model 2": kr2lm})
    
    img_stream.set_config(folder_path=data_path, fps=30)
    od_stream.set_config(model_id=args.model_id)
    fr_stream.set_config(model_id="vggface2") # microsoft/resnet-101
    lpr_stream.set_config(model_id="./pytorch-licenseplate-segmentation/model_v2.pth")
    cap_stream.set_config(model_id="microsoft/git-base")
    kr_stream.set_config(model_id="gpt2")
    lm_stream.set_config(model_id="gpt2")
    
    processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, lm_stream, queue_watch]
    queues = [img2od, od2fr, od2lpr, od2cap, fr2kr, lpr2kr, kr2lm, cap2lm]
    queue_names = ["img2od", "od2fr", "od2lpr", "od2cap", "fr2kr", "lpr2kr", "kr2lm", "cap2lm"]
    

    logger.info("Starting processes")
    for p in processes:
        p.start()
    
    try:
        logger.info("Pipeline running")
        
        for p in processes:
            p.join()
        
        logger.info("All processes completed")
        
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")
    finally:
        clean_up(processes, queues)
        logger.info("Pipeline shutdown complete")