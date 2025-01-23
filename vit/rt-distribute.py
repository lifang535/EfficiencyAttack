from transformers import AutoImageProcessor
from transformers import VitDetConfig, VitDetModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
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
import random
# import overload_attack
# import adaptive_attack
from datetime import datetime
# import stra_attack
import time
import phantom_attack
from transformers import AutoImageProcessor
from transformers import VitDetConfig, VitDetModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import RTDetrConfig, RTDetrModel
import torchvision.transforms as transforms
import torch
import torch.multiprocessing as mp
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
from datetime import datetime
import os
import random
import time

def setup_gpu_device(gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    return device

def process_batch(gpu_id, data_batch, epoch_num, save_dir, algo_name, target_cls_idx, results_dict):
    try:
        device = setup_gpu_device(gpu_id)
        print(f"=============================RUNNING {gpu_id} RUNNING==============================")
        # Initialize model and processor for this process
        image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        from transformers import RTDetrConfig, RTDetrForObjectDetection
        model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").to(device)

        model.eval()
        
        if algo_name == "phantom":
        # Create Phantom Attack instance for this batch
            phantom = phantom_attack.PhantomAttack(
                model, 
                image_processor,
                image_list=data_batch,
                image_name_list=None,
                img_size=None,
                epochs=epoch_num,
                device=device
            )
        
            phantom.run()
            results_dict.update(phantom.results_dict)
            
        if algo_name == "overload":
            import overload_attack
        # Create Phantom Attack instance for this batch
            overload = overload_attack.OverloadAttack(
                model, 
                image_processor,
                image_list=data_batch,
                image_name_list=None,
                img_size=None,
                epochs=epoch_num,
                device=device
            )
        
            overload.run()
            results_dict.update(overload.results_dict)
            
        if algo_name == "ada":
            import adaptive_attack
            ada = adaptive_attack.SingleAttack(model, image_processor,
                                       image_list=data_batch,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=epoch_num,
                                        cls_idx = target_cls_idx,
                                        device=device)
            ada.run()
            results_dict.update(ada.results_dict)

        if algo_name == "slow":
            import stra_attack
            slow = stra_attack.StraAttack(model, image_processor,
                                        image_list=data_batch,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=epoch_num,
                                        device=device)
            slow.run()
            results_dict.update(slow.results_dict)  
            
    except Exception as e:
        print(f"Error in GPU {gpu_id} process: {e}")
        
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            model.cpu()
            del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_id)

def parallel_attack(coco_data, num_gpus, epoch_num, algo_name, target_cls_idx=None, save_dir=None):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Split data into batches for each GPU
    batch_size = len(coco_data) // num_gpus
    data_batches = [
        coco_data.select(range(i * batch_size, (i + 1) * batch_size))
        for i in range(num_gpus)
    ]
    
    # Handle remaining samples
    if len(coco_data) % num_gpus != 0:
        remaining = coco_data.select(range(num_gpus * batch_size, len(coco_data)))
        data_batches[-1] = data_batches[-1].concatenate(remaining)
    
    # Create a Manager for sharing data between processes
    manager = mp.Manager()
    results_dict = manager.dict()
    
    try:
        # Create processes
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=process_batch,
                args=(gpu_id, data_batches[gpu_id], epoch_num, save_dir, algo_name, target_cls_idx, results_dict)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Convert manager dict to regular dict
        combined_results = dict(results_dict)
        
        return combined_results
        
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise e
        
    finally:
        # Clean up manager
        manager.shutdown()

def init_process():
    mp.set_start_method('spawn', force=True)

if __name__ == "__main__":
    init_process()
    util.set_all_seeds(0)

    parser = argparse.ArgumentParser(description="DETR hyperparam setup")
    parser.add_argument("--epoch_num", type=int, default=200)
    parser.add_argument("--val_size", type=int, choices=range(1, 4952), default=1000, 
                        help="An integer in the range 1-4952 (inclusive)")
    parser.add_argument("--algo_name", type=str, default="infer")
    parser.add_argument("--pipeline_name", type=str, default=None)
    parser.add_argument("--target_cls_idx", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=None, 
                        help="Number of GPUs to use. Defaults to all available GPUs.")
    args = parser.parse_args()

    # Determine number of available GPUs if not specified
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count()
    
    print(f"Running on {args.num_gpus} GPUs")

    # Set up MS COCO 2017
    coco_data = load_dataset("detection-datasets/coco", split="val")
    random_indices = random.sample(range(len(coco_data)), args.val_size)
    coco_data = coco_data.select(random_indices)

    if args.algo_name == "phantom":
        # Create results directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('results', f'phantom_attack_{timestamp}')
        
        results_dict = parallel_phantom_attack(coco_data, args.num_gpus, args.epoch_num, save_dir)
        
        # Save final results
        results_file = os.path.join(save_dir, 'rt_final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Results saved to {results_file}")
        
    if args.algo_name == "ada":
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('results', f'ada_attack_{timestamp}')
        
        results_dict = parallel_attack(coco_data, args.num_gpus, args.epoch_num, args.algo_name, args.target_cls_idx, save_dir)
        
        results_file = os.path.join(save_dir, 'rt-final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Results saved to {results_file}")
        
    if args.algo_name == "overload":
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('results', f'overload_attack_{timestamp}')
        
        results_dict = parallel_attack(coco_data, args.num_gpus, args.epoch_num, args.algo_name, args.target_cls_idx, save_dir)
        
        results_file = os.path.join(save_dir, 'rt-final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Results saved to {results_file}")


    if args.algo_name == "slow":
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('results', f'overload_attack_{timestamp}')
        
        results_dict = parallel_attack(coco_data, args.num_gpus, args.epoch_num, args.algo_name, args.target_cls_idx, save_dir)
        
        results_file = os.path.join(save_dir, 'rt-final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"Results saved to {results_file}")
