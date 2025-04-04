from cmpnt.c1_img import imgStream
from cmpnt.c2_det import odStream
from cmpnt.c3_fr import frStream
from cmpnt.c4_lpr import lprStream
from cmpnt.c5_cap import capStream
from cmpnt.c6_kr import krStream
from cmpnt.c7_udp import udpStream
from cleanup import cleanup_resources
import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process, Event
import logging
import argparse
import json
import glob
from flops import summary

parser = argparse.ArgumentParser(description="Traffic Monitoring Pipeline")
parser.add_argument("--model_id", type=int, default=0, help="Model ID for object detection")
parser.add_argument('--algorithm', type=str, default=None, choices=["overload", 
                                                                    "slowtrack", 
                                                                    "phantom", 
                                                                    "teaspoon", 
                                                                    "teastatic"], help="algorithm not found")
parser.add_argument('--target_idx', type=int, nargs='+', default=None, help="List of numbers, unavailable for baseline")
parser.add_argument("--ps_path", type=str, default="./profile", help="Path to save profile data")
args = parser.parse_args()

if args.target_idx:
    target_indices = ('_'.join(map(str, args.target_idx)))
else:
    target_indices = "none"
    
base_dir = "../saved"
model_id = args.model_id
algorithm = args.algorithm

if algorithm is None:
    input_dir = "./test_src"
    ps_path = "./test_profile"
else:
    ps_path = os.path.join(args.ps_path, f"model_{args.model_id}", f"{args.algorithm}_tgt_{target_indices}")
    input_dir = os.path.join(base_dir, f"model_{args.model_id}", f"{args.algorithm}_tgt_{target_indices}")
    
# print(f"Input directory: {input_dir}")
# print(f"Profile save path: {ps_path}")

os.makedirs(ps_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

            
if __name__ == "__main__":
    date_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    title = "Running: Traffic Monitoring Pipeline"
    subtitle = f"Algorithm: {algorithm}, Model ID: {model_id}"
    subsubtitle = f"Target Indices: {target_indices}"
    
    print("\n\n" + "=" * 80)
    print("|" + " ".center(78) + "|")
    print("|" + title.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + subtitle.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + subsubtitle.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("|" + date_time.center(78) + "|")
    print("|" + " ".center(78) + "|")
    print("=" * 80 + "\n\n")
    
    start_time = time.perf_counter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.synchronize()

    mp.set_start_method("spawn")
    
    ms = 1000
    
    img2od_queue = Queue(maxsize=ms)
    od2fr_queue = Queue(maxsize=ms)
    od2lpr_queue = Queue(maxsize=ms)
    od2cap_queue = Queue(maxsize=ms)
    fr2kr_queue = Queue(maxsize=ms)
    lpr2kr_queue = Queue(maxsize=ms)
    cap2udp_queue = Queue(maxsize=ms)
    kr2udp_queue = Queue(maxsize=ms)
    
    img_stream = imgStream(img2od_queue, device)
    od_stream = odStream(img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device)
    fr_stream = frStream(od2fr_queue, fr2kr_queue, device)
    lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, device)
    cap_stream = capStream(od2cap_queue, cap2udp_queue, device)
    kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2udp_queue, device)
    udp_Stream = udpStream(cap2udp_queue, kr2udp_queue, device)
    
    img_stream.set_config(src_folder_path = input_dir, fps = 30, profile_save_path=ps_path)
    od_stream.set_config(model_id=model_id, profile_save_path=ps_path)
    fr_stream.set_config(profile_save_path=ps_path)
    lpr_stream.set_config(profile_save_path=ps_path)
    cap_stream.set_config(profile_save_path=ps_path)
    kr_stream.set_config(embedding_path="./face_embeddings", profile_save_path=ps_path)
    udp_Stream.set_config(profile_save_path=ps_path)

    
    processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, udp_Stream]
    queues = [img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, fr2kr_queue, lpr2kr_queue, cap2udp_queue, kr2udp_queue]
    queue_names = ["img2od", "od2fr", "od2lpr", "od2cap", "fr2kr", "lpr2kr", "cap2udp", "kr2udp"]
    
    from queue_watch import QueueWatch
    queue_watcher = QueueWatch(queues, queue_names, processes)
    queue_watcher.set_config()
    queue_watcher.start()
    
    for p in processes:
        p.start()
        time.sleep(0.1)  # Optional: small delay to ensure all processes start properly
    
    try:
        for p in processes:
            p.join()
            time.sleep(0.1)  # Optional: small delay to ensure all processes finish properly
        queue_watcher.join()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user in MAIN process")
    except Exception as e:
        logger.error(f"Error in MAIN process: {str(e)}")
    finally:
        logger.info("Reach FINALLY block in MAIN process")
        cleanup_resources(processes, queues)
        logger.info("Cleaned up resources in MAIN process")
        
    end_time = time.perf_counter()
    
    summary(ps_path, start_time, end_time)