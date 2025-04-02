import numpy as np
import sys
from pathlib import Path
import os
from tabulate import tabulate
import logging
import psutil
import functools
from typing import Dict, Set, Any, Callable, Optional, List
import threading
import json
import inspect
import torch
import subprocess
import re
from datetime import datetime
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
from torch.profiler import profile, record_function, ProfilerActivity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append("./components")
output_folder = "./profile"


def FLOPs_DECORATOR(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs(output_folder, exist_ok=True)
        save_path = f"{output_folder}/{func.__qualname__}_profile" if output_folder else f"{func.__qualname__}_profile"
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                profile_memory=True,
                record_shapes=True
            ) as prof:
                with record_function(func.__qualname__):
                    result = func(*args, **kwargs)
            # print("\n" + "i am here " * 10 + "\n")
                    
            print(prof.key_averages().table())
                
            cpu_flops = 0
            cuda_flops = 0
            operations_data = []
            
            cpu_time_total = None
            cuda_time_total = None
            
            table_str = str(prof.key_averages().table(sort_by="cuda_time_total"))
            table_rows = table_str.split('\n')

        except Exception as e:
            logging.error(f"Error during profiling: {e}")
            return None
        
        
        for row in table_rows:
            if "Self CPU time total" in row:
                parts = row.split(":")
                if len(parts) > 1:
                    cpu_time_total = parts[1].strip()
                    # cpu_time_total = sum(evt.self_cpu_time_total for evt in prof.key_averages())
            elif "Self CUDA time total" in row:
                parts = row.split(":")
                if len(parts) > 1:
                    cuda_time_total = parts[1].strip()
                    # cuda_time_total = sum(evt.self_cuda_time_total for evt in prof.key_averages())
                    
        header_parts = None
        for row in table_rows:
            if "Total MFLOPs" in row:
                header_parts = row.split()
                break
            
        
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
                
        with open(save_path + ".txt", "w") as f:
            # Write summary metrics at the top for quick reference
            f.write(f"Self CPU time total: {cpu_time_total} \n")
            f.write(f"Self CUDA time total: {cuda_time_total} \n")
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
        
        with open(save_path + ".json", 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"{func.__qualname__} profile saved")
        return result
    return wrapper

def write_profile(prof, save_path):
    cpu_flops = 0
    cuda_flops = 0
    operations_data = []
    
    cpu_time_total = None
    cuda_time_total = None
    
    table_str = str(prof.key_averages().table(sort_by="cuda_time_total"))
    table_rows = table_str.split('\n')

    for row in table_rows:
        if "Self CPU time total" in row:
            parts = row.split(":")
            if len(parts) > 1:
                cpu_time_total = parts[1].strip()
                # cpu_time_total = sum(evt.self_cpu_time_total for evt in prof.key_averages())
        elif "Self CUDA time total" in row:
            parts = row.split(":")
            if len(parts) > 1:
                cuda_time_total = parts[1].strip()
                # cuda_time_total = sum(evt.self_cuda_time_total for evt in prof.key_averages())
                
    header_parts = None
    for row in table_rows:
        if "Total MFLOPs" in row:
            header_parts = row.split()
            break
        

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
            
    with open(save_path + ".txt", "w") as f:
            # Write summary metrics at the top for quick reference
            f.write(f"Self CPU time total: {cpu_time_total} \n")
            f.write(f"Self CUDA time total: {cuda_time_total} \n")
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

    with open(save_path + ".json", 'w') as f:
        json.dump(profile_data, f, indent=2)

    logger.info(f"{save_path} profile saved")
    
if __name__ == "__main__":
    pass