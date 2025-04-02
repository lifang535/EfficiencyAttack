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


input_dir = "./test_src"
output_dir = "./test_src"
hf_model_id = 0

ps_path = "./profile"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.synchronize()

    mp.set_start_method("spawn")
    
    img2od_queue = Queue()
    od2fr_queue = Queue()
    od2lpr_queue = Queue()
    od2cap_queue = Queue()
    fr2kr_queue = Queue()
    lpr2kr_queue = Queue()
    cap2udp_queue = Queue()
    kr2udp_queue = Queue()
    
    img_stream = imgStream(img2od_queue, device)
    od_stream = odStream(img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device)
    fr_stream = frStream(od2fr_queue, fr2kr_queue, device)
    lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, device)
    cap_stream = capStream(od2cap_queue, cap2udp_queue, device)
    kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2udp_queue, device)
    udp_Stream = udpStream(cap2udp_queue, kr2udp_queue, device)
    
    img_stream.set_config(src_folder_path = input_dir, fps = 30, profile_save_path=ps_path)
    od_stream.set_config(model_id=hf_model_id, profile_save_path=ps_path)
    fr_stream.set_config(profile_save_path=ps_path)
    lpr_stream.set_config(profile_save_path=ps_path)
    cap_stream.set_config(profile_save_path=ps_path)
    kr_stream.set_config(embedding_path="./face_embeddings", profile_save_path=ps_path)
    udp_Stream.set_config(profile_save_path=ps_path)

    
    processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, udp_Stream]
    queues = [img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, fr2kr_queue, lpr2kr_queue, cap2udp_queue, kr2udp_queue]
    queue_names = ["img2od", "od2fr", "od2lpr", "od2cap", "fr2kr", "lpr2kr", "cap2udp", "kr2udp"]
    
    from queue_watch import QueueWatch
    queue_watcher = QueueWatch(queues, queue_names)
    queue_watcher.set_config()
    queue_watcher.start()
    
    for p in processes:
        p.start()
        time.sleep(0.1)  # Optional: small delay to ensure all processes start properly
    
    try:
        queue_watcher.join()
        for p in processes:
            p.join()
            time.sleep(0.1)  # Optional: small delay to ensure all processes finish properly
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user in MAIN process")
    except Exception as e:
        logger.error(f"Error in MAIN process: {str(e)}")
    finally:
        logger.info("Reach FINALLY block in MAIN process")
        cleanup_resources(processes, queues)
        logger.info("Cleaned up resources in MAIN process")