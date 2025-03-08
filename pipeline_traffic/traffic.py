from image_streaming import imgStream
from object_detection import odStream
from face_recognition import frStream
from license_plate_recognition import lprStream
from image_captioning import capStream
from knowledge_retrieval import krStream
from language_model import lmStream
from pipeline_utils import clean_up


from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
sys.path.append("../")
import os
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model_zoo import load_from_pretrained

import argparse
parser = argparse.ArgumentParser(description="Traffic pipeline")
parser.add_argument("--model_id", type=int, default=None)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--data_path", type=str, default="./sample_data")
args = parser.parse_args()

if args.algorithm or args.model_id:
    pass
else:
    raise ValueError("Please specify an algorithm and a model id")

from dotenv import load_dotenv, set_key, dotenv_values
dotenv_path = ".env"  
load_dotenv(dotenv_path)
set_key(dotenv_path, "SESSION_ID", f"{args.model_id}_{args.algorithm}")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
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
    
    img_stream.set_config(folder_path=args.data_path, fps=30)
    od_stream.set_config(model_id=args.model_id)
    fr_stream.set_config(model_id="vggface2") # microsoft/resnet-101
    lpr_stream.set_config(model_id="./pytorch-licenseplate-segmentation/model_v2.pth")
    cap_stream.set_config(model_id="microsoft/git-base")
    kr_stream.set_config(model_id="gpt2")
    lm_stream.set_config(model_id="gpt2")
    
    processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, lm_stream]
    queues = [img2od, od2fr, od2lpr, od2cap, fr2kr, lpr2kr, kr2lm, cap2lm]
    

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
    
    try:
        logger.info("Pipeline running")
        
        # Wait for processes to complete
        logger.info("Waiting for image stream to complete")
        img_stream.join()
        logger.info("Image stream completed")
        
        # Give OD stream time to process any remaining items
        logger.info("Waiting for OD stream to complete")
        od_stream.join(timeout=36000.0)
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