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
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueueWatch(Process):
    def __init__(self, queue_dict, device=None):
        super().__init__(name="QueueWatch")
        self.queue_dict = queue_dict
        self.stop_event = Event()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_queues = len(queue_dict)
        self.pbars = {}
        
    def run(self):
        # Set up proper terminal handling for tqdm
        for i, (queue_name, queue) in enumerate(self.queue_dict.items()):
            self.pbars[queue_name] = tqdm(
                total=0,  # We're just counting, not tracking progress to a goal
                desc=f"{queue_name}",
                position=i,
                leave=True,
                bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                # Disable the progress bar itself but keep the information
                ncols=100
            )
        
        try:
            while not self.stop_event.is_set():
                # Update each progress bar with current queue size
                for queue_name, queue in self.queue_dict.items():
                    try:
                        size = queue.qsize()
                        pbar = self.pbars[queue_name]
                        # Instead of just updating n, reset total and n for smoother updates
                        pbar.total = max(size, 1)  # Avoid division by zero
                        pbar.n = size
                        pbar.refresh()
                    except (EOFError, BrokenPipeError, NotImplementedError):
                        # Handle various queue-related errors
                        pass
                
                # Sleep longer to reduce update frequency
                time.sleep(0.5)
        finally:
            # Ensure proper cleanup of all progress bars
            for i, pbar in enumerate(self.pbars.values()):
                pbar.clear()
                pbar.close()
    
    def stop(self):
        self.stop_event.set()
        self.join(timeout=1.0)  # Give it time to clean up