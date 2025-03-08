from multiprocessing import Process, Queue, Event
from queue import Empty
import multiprocessing as mp
import glob
import time
import torch
import numpy as np
import sys
sys.path.append("../")
import functools
import os
import datetime
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, ResNetForImageClassification, AutoModelForImageClassification
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model_zoo import load_from_pretrained
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()
SESSION_ID = os.getenv("SESSION_ID")
# import traffic

# SESSION_ID = traffic.SESSION_ID
LOG_DIR = f"profile/{SESSION_ID}"

os.makedirs(LOG_DIR, exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_flops_decorator(func, save_path=None):

    import functools
    import json
    import os
    from torch.profiler import profile, record_function, ProfilerActivity
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if save_path is None:
            file_path = f"{LOG_DIR}/{func.__qualname__}_profile.json"
        else:
            file_path = save_path
            
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            profile_memory=True,
            record_shapes=True
        ) as prof:
            with record_function(func.__qualname__):
                result = func(*args, **kwargs)
        
        cpu_flops = 0
        cuda_flops = 0
        operations_data = []
        
        cpu_time_total = None
        cuda_time_total = None
        
        table_str = str(prof.key_averages().table(sort_by="cuda_time_total"))
        table_rows = table_str.split('\n')
        
        # for row in table_rows:
        #     if "Self CPU time total" in row:
        #         parts = row.split(":")
        #         if len(parts) > 1:
        #             cpu_time_total = parts[1].strip()
        #     elif "Self CUDA time total" in row:
        #         parts = row.split(":")
        #         if len(parts) > 1:
        #             cuda_time_total = parts[1].strip()
        


        header_parts = None
        for row in table_rows:
            if "Total MFLOPs" in row:
                header_parts = row.split()
                break
        
        if not header_parts:
            logger.warning(f"Could not find 'Total MFLOPs' column in profiler output of {func.__qualname__}")
            profile_data = {
                "timing": {
                    "self_cpu_time_total": cpu_time_total,
                    "self_cuda_time_total": cuda_time_total
                },
                "operations": [],
                "flops": {
                    "cpu_flops": 0.0,
                    "cuda_flops": 0.0,
                    "total_flops": 0.0
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
                
            # print(f"Profile data saved to {file_path}")
            return result
            
        try:
            mflops_col_idx = header_parts.index("Total") + 1
        except ValueError:
            logger.info(f"Warning: Could not determine MFLOPs column index {func.__qualname__}")
            return result
        
        mflops_data = []
        for row in table_rows:
            if len(row.strip()) == 0 or row.startswith('----'):
                continue
                
            columns = row.split()
            if len(columns) < 2:
                continue
            
            op_name = None
            percentage = None
            duration = None
            
            name_parts = []
            for part in columns[:3]:
                if '%' in part:  
                    break
                name_parts.append(part)
            
            if name_parts:
                op_name = ' '.join(name_parts)
            
            for part in columns:
                if '%' in part and part[-1] == '%': 
                    percentage = part
                    break
            
            for part in columns:
                if any(unit in part for unit in ['us', 'ms', 's']) and any(c.isdigit() for c in part):
                    duration = part
                    break
                
            last_col = columns[-1]
            if last_col != '--' and any(c.isdigit() for c in last_col):
                try:
                    if not op_name:
                        op_name = ' '.join(columns[:min(3, len(columns))])
                    
                    mflops_val = float(last_col)
                    
                    is_cuda_op = False
                    
                    cuda_related = ['cuda', 'gpu', 'volta', 'cudnn', 'device', 'gemv', 'sgemm']
                    if any(term in op_name.lower() for term in cuda_related):
                        is_cuda_op = True
                    
                    common_gpu_ops = ['conv2d', 'batch_norm', 'pool2d', 'relu', 'addmm']
                    if any(op in op_name.lower() for op in common_gpu_ops):
                        is_cuda_op = True
                    
                    if is_cuda_op:
                        cuda_flops += mflops_val * 1e6
                    else:
                        cpu_flops += mflops_val * 1e6
                        
                    operation = {
                        "name": op_name,
                        "percentage": percentage,
                        "duration": duration,
                        "mflops": mflops_val,
                        "device": "CUDA" if is_cuda_op else "CPU"
                    }
                    operations_data.append(operation)
                    
                    mflops_data.append((op_name, mflops_val, is_cuda_op))
                except ValueError:
                    pass
                
        with open(f"{LOG_DIR}/{func.__qualname__}_profile.txt", "w") as f:
                # Write summary metrics at the top for quick reference
                f.write(f"Self CPU time total: {cpu_time_total} ms\n")
                f.write(f"Self CUDA time total: {cuda_time_total} ms\n")
                f.write("\n--- Full Profiling Table ---\n")
                # Write the full table
                f.write(table_str)
                
                # Add FLOPS section
                f.write("\n\n--- MFLOPs Data ---\n")
                mflops_table = []
                for op, mflops, is_cuda in mflops_data:
                    device = "CUDA" if is_cuda else "CPU"
                    mflops_table.append([op, f"{mflops:.2f}", device])
                
                f.write(tabulate(mflops_table, headers=["Operation", "MFLOPs", "Device"], tablefmt="grid"))
                
                # Summary FLOPs
                f.write("\n\n--- FLOPs Summary ---\n")
                flops_summary = [
                    ["CPU FLOPs", f"{cpu_flops:.2f}"],
                    ["CUDA FLOPs", f"{cuda_flops:.2f}"],
                    ["Total FLOPs", f"{cpu_flops + cuda_flops:.2f}"]
                ]
                f.write(tabulate(flops_summary, headers=["Metric", "Value"], tablefmt="grid"))
            
            
        # print(f"\n===== Profile for {func.__qualname__} =====")
        # print(table_str)
        
        # print("\nExtracted MFLOPs data:")
        # for op, mflops, is_cuda in mflops_data:
        #     device = "CUDA" if is_cuda else "CPU"
        #     print(f"{op}: {mflops} MFLOPs ({device})")
            
        # print(f"\nCPU FLOPs: {cpu_flops:.2f}")
        # print(f"CUDA FLOPs: {cuda_flops:.2f}")
        # print(f"Total FLOPs: {cpu_flops + cuda_flops:.2f}")
        
        profile_data = {
            "timing": {
                "self_cpu_time_total": cpu_time_total,
                "self_cuda_time_total": cuda_time_total
            },
            "operations": operations_data,
            "flops": {
                "cpu_flops": cpu_flops,
                "cuda_flops": cuda_flops,
                "total_flops": cpu_flops + cuda_flops
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
            
        # print(f"\nProfile data saved to {file_path}")
        # print(profile_data)
        return result
    return wrapper


def clean_up(processes, queues):
    """Clean up processes and queues"""
    logger.info("Cleaning up resources")
    
    # Signal processes to stop
    for p in processes:
        if hasattr(p, 'shutdown') and p.is_alive():
            logger.info(f"Shutting down {p.name}")
            p.shutdown()
    
    # Wait w moment for graceful shutdown
    time.sleep(1.0)
    
    # Force terminate if needed
    for p in processes:
        if p.is_alive():
            logger.info(f"Terminating {p.name}")
            p.terminate()
            p.join(timeout=1.0)
            if p.is_alive():
                logger.warning(f"Process {p.name} did not terminate, killing")
                p.kill()
    
    # Drain queues
    for q in queues:
        try:
            while not q.empty():
                _ = q.get_nowait()
        except:
            pass
    
    # Final cleanup
    torch.cuda.empty_cache()
    logger.info("Cleanup complete")
    
    
