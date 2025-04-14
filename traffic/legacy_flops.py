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
from torch.profiler import schedule
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
            
    # with open(save_path + ".txt", "w") as f:
    #         # Write summary metrics at the top for quick reference
    #         f.write(f"Self CPU time total: {cpu_time_total} \n")
    #         f.write(f"Self CUDA time total: {cuda_time_total} \n")
    #         f.write("\n--- Full Profiling Table ---\n")
    #         # Write the full table
    #         f.write(table_str)
            
    #         # Add FLOPS section
    #         f.write("\n\n--- MFLOPs Data ---\n")
    #         mflops_table = []
    #         for op, mflops, is_cuda in mflops_data:
    #             device = "CUDA" if is_cuda else "CPU"
    #             mflops_table.append([op, f"{mflops:.2f}", device])
            
    #         f.write(tabulate(mflops_table, headers=["Operation", "MFLOPs", "Device"], tablefmt="grid"))
            
    #         # Summary FLOPs
    #         f.write("\n\n--- FLOPs Summary ---\n")
    #         flops_summary = [
    #             ["CPU FLOPs", f"{cpu_flops:.2f}"],
    #             ["CUDA FLOPs", f"{cuda_flops:.2f}"],
    #             ["Total FLOPs", f"{cpu_flops + cuda_flops:.2f}"]
    #         ]
    #         f.write(tabulate(flops_summary, headers=["Metric", "Value"], tablefmt="grid"))

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
    
def write_profile_lt(prof, save_path):
    cpu_flops = 0
    cuda_flops = 0
    
    # Directly iterate through profile events
    for evt in prof.key_averages():
        # Skip events without flops information
        if not hasattr(evt, 'flops') or evt.flops == 0:
            continue
            
        # Convert MFLOPs to FLOPs
        flops_val = float(evt.flops)
        
        # Check if the operation is CUDA-related
        is_cuda_op = False
        op_name = evt.key
        
        # Check for CUDA operations
        cuda_related = ['cuda', 'gpu', 'volta', 'cudnn', 'device', 'gemv', 'sgemm']
        if any(term in op_name.lower() for term in cuda_related):
            is_cuda_op = True
        
        # Common GPU operations
        common_gpu_ops = ['conv2d', 'batch_norm', 'pool2d', 'relu', 'addmm']
        if any(op in op_name.lower() for op in common_gpu_ops):
            is_cuda_op = True
        
        # Add to appropriate counter
        if is_cuda_op:
            cuda_flops += flops_val
        else:
            cpu_flops += flops_val
    
    # Get total times
    cpu_time_total = sum(evt.self_cpu_time_total for evt in prof.key_averages())
    cuda_time_total = sum(evt.self_cuda_time_total for evt in prof.key_averages())
    
    # Create summary data
    profile_data = {
        "timing": {
            "self_cpu_time_total": f"{cpu_time_total:.2f} us",
            "self_cuda_time_total": f"{cuda_time_total:.2f} us" if cuda_time_total else "N/A"
        },
        "flops": {
            "cpu_flops": cpu_flops,
            "cuda_flops": cuda_flops,
            "total_flops": cpu_flops + cuda_flops
        }
    }

    with open(save_path + ".json", 'w') as f:
        json.dump(profile_data, f, indent=2)

    logger.info(f"{save_path} profile saved")

import glob
import re

def summary(ps_path, start_time, end_time):
    
    pipe_cpu_time = 0
    pipe_cuda_time = 0
    pipe_time = end_time - start_time

    pipe_cpu_flops = 0
    pipe_cuda_flops = 0
    pipe_flops = 0
    
    json_profiles = glob.glob(os.path.join(ps_path, "*.json"))

    if not json_profiles:
        print(f"No JSON files found in {json_profiles}")
        return
    
    for json_file in json_profiles:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

                if "timing" in data:
                    # Convert time strings to seconds (assuming format like "560.700s")
                    cpu_time_str = data["timing"].get("self_cpu_time_total", "0s")
                    cuda_time_str = data["timing"].get("self_cuda_time_total", "0s")
                    
                    cpu_time = convert_time_to_seconds(cpu_time_str)
                    cuda_time = convert_time_to_seconds(cuda_time_str)
                    
                    pipe_cpu_time += cpu_time
                    pipe_cuda_time += cuda_time
                
                # Extract flops information
                if "flops" in data:
                    pipe_cpu_flops += data["flops"].get("cpu_flops", 0)
                    pipe_cuda_flops += data["flops"].get("cuda_flops", 0)
                    pipe_flops += data["flops"].get("total_flops", 0)
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Create summary JSON
    summary = {
        "timing": {
            "self_cpu_time_total": f"{pipe_cpu_time:.3f}s",
            "self_cuda_time_total": f"{pipe_cuda_time:.3f}s",
            "total_time": f"{pipe_time:.3f}s"
        },
        "flops": {
            "cpu_flops": pipe_cpu_flops,
            "cuda_flops": pipe_cuda_flops,
            "total_flops": pipe_flops
        }
    }
    
    # Write summary to a new JSON file
    output_path = os.path.join(ps_path, "SUMMARY.json")
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary written to {output_path}")
    

def convert_time_to_seconds(time_str):
    """Convert a time string with units to seconds."""
    if time_str is None:
        return 0
    
    # Convert to string in case it's not already
    time_str = str(time_str)
    
    # Define conversion factors for different time units to seconds
    unit_to_seconds = {
        's': 1,
        'ms': 0.001,
        'us': 0.000001,
        'µs': 0.000001,  # Unicode micro symbol
        'ns': 0.000000001,
        'm': 60,         # minutes
        'h': 3600        # hours
    }
    
    # Extract number and unit using regex
    match = re.match(r'([\d.]+)([a-zµ]+)', time_str)
    if match:
        value, unit = match.groups()
        multiplier = unit_to_seconds.get(unit, 1)  # Default to seconds if unknown unit
        return float(value) * multiplier
    
    # If no unit is specified or pattern doesn't match, assume seconds
    try:
        return float(time_str.strip())
    except ValueError:
        print(f"Warning: Could not parse time value '{time_str}', treating as 0")
        return 0
    

if __name__ == "__main__":
    pass