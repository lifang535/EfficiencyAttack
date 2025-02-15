from datasets import load_dataset
import dataset
from utils import set_all_seeds
import torch
import random 
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import utils 
import os 
import pdb
from model_zoo import load_from_pretrained

set_all_seeds(0)


parser = argparse.ArgumentParser(description="RT-DETR attack setup")
parser.add_argument('--target_idx', type=int, nargs='+', default=None, help='List of numbers')
parser.add_argument("--val_size", type=int, default=1000, help="An integer in the range 1-4952 (inclusive)")
parser.add_argument("--it_num", type=int, default=200, help="iteration num per attack")
parser.add_argument('--algorithm', type=str, default=None, choices=["overload", "slowtrack", "phantom", "teaspoon"],help="choose from [overload, slowtrack, phantom, teaspoon]")
parser.add_argument('--model_id', type=int, default=None, choices=[0,1,2], help="0: PekingU/rtdetr_r50vd, \
                                                                                 1: PekingU/rtdetr_r50vd_coco_o365, \
                                                                                 2: PekingU/rtdetr_v2_r50vd")
parser.add_argument('--output_dir', type=str, default="../results", help="specify where to save the results")
args = parser.parse_args()


    
    
def process_batch(
        gpu_id,
        data_batch,
        dir
    ):

    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        model, image_processor = load_from_pretrained(args.model_id)
        model = model.to(device).eval()
        
        print(f"=============================RUNNING {gpu_id} RUNNING==============================")
        if args.algorithm == "overload":
            from overload import Overload
            class_name = Overload
        elif args.algorithm == "phantom":
            from phantom import Phantom
            class_name = Phantom
            pass
        elif args.algorithm == "slowtrack":
            from slow import SlowTrack
            class_name = SlowTrack
            pass
        elif args.algorithm == "teaspoon":
            from teaspoon import teaspoon
            class_name = teaspoon
            pass
        else:
            raise ValueError("algorithm not implemented")

        
        instance = class_name(
            model = model,
            image_processor = image_processor,
            it_num = args.it_num,
            conf_thres = 0.25,
            target_idx = args.target_idx,
            output_dir = dir,
            device = device
        )
        
        for index, example in tqdm(enumerate(data_batch), total=data_batch.__len__(), desc=f"running: {args.algorithm}"):
            image_id, image, width, height, bbox_id, category, gt_boxes, area = utils.parse_example(example)
            instance.run_attack(image, image_id)
            
    except Exception as e:
        print(f"Error in GPU {gpu_id} process: {e}")
        
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            model.cpu()
            del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_id)


def parallel(coco_data, num_gpus):
    if args.target_idx:
        target_indices = ('_'.join(map(str, args.target_idx)))
    else:
        target_indices = "none"
    dir = os.path.join(args.output_dir, f"model_{args.model_id}", \
                                        f"{args.algorithm}_tgt_{target_indices}")
    os.makedirs(dir, exist_ok=True)
    batch_size = len(coco_data) // num_gpus
    data_batches = [
        coco_data.select(range(i * batch_size, (i + 1) * batch_size))
        for i in range(num_gpus)
    ]
    if len(coco_data) % num_gpus != 0:
        remaining = coco_data.select(range(num_gpus * batch_size, len(coco_data)))
        data_batches[-1] = data_batches[-1].concatenate(remaining)
            
    try:
        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=process_batch,
                args=(
                    gpu_id,
                    data_batches[gpu_id],
                    dir
                )
            )
            
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        for p in processes:
            p.terminate() 


if __name__ == "__main__":
    if not torch.cuda.is_available():
        ValueError("cuda device not found!")
        
    gpu_count = torch.cuda.device_count()
    
    coco_data = load_dataset("detection-datasets/coco", split="val")
    random_indices = random.sample(range(len(coco_data)), args.val_size)
    coco_data = coco_data.select(random_indices)
    
    mp.set_start_method('spawn', force=True)
    
    parallel(coco_data, gpu_count)
    