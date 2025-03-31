from cmpnt.c1_img import imgStream
from cmpnt.c2_det import odStream
from cmpnt.c3_fr import frStream
from cmpnt.c4_lpr import lprStream
from cmpnt.c5_cap import capStream
from cmpnt.c6_kr import krStream
from cmpnt.c7_lm import lmStream
from cleanup import cleanup_resources
import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Queue

input_dir = "./test_src"
output_dir = "./test_src"
hf_model_id = 0

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
    cap2lm_queue = Queue()
    kr2lm_queue = Queue()
    
    img_stream = imgStream(img2od_queue, device)
    od_stream = odStream(img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, device)
    fr_stream = frStream(od2fr_queue, fr2kr_queue, device)
    lpr_stream = lprStream(od2lpr_queue, lpr2kr_queue, device)
    cap_stream = capStream(od2cap_queue, cap2lm_queue, device)
    kr_stream = krStream(fr2kr_queue, lpr2kr_queue, kr2lm_queue, device)
    lm_stream = lmStream(cap2lm_queue, kr2lm_queue, device)
    
    img_stream.set_config(src_folder_path = input_dir, fps = 30)
    od_stream.set_config(model_id=hf_model_id)
    fr_stream.set_config()
    lpr_stream.set_config()
    cap_stream.set_config()
    lm_stream.set_config()

    
    processes = [img_stream, od_stream, fr_stream, lpr_stream, cap_stream, kr_stream, lm_stream]
    queues = [img2od_queue, od2fr_queue, od2lpr_queue, od2cap_queue, fr2kr_queue, lpr2kr_queue, cap2lm_queue, kr2lm_queue]
    
    for p in processes:
        p.start()
        time.sleep(0.1)  # Optional: small delay to ensure all processes start properly
    
    try:
        for p in processes:
            p.join(timeout=5)
            time.sleep(0.1)  # Optional: small delay to ensure all processes finish properly
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pass
        cleanup_resources(processes, queues)