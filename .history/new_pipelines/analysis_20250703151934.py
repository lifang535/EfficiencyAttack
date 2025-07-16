import os
import sys
import pdb
import glob
import json
import random
import numpy as np
from tqdm import tqdm
sys.path.append("../")
import utils; utils.set_all_seeds(0)




def analyze_distribution(values, label):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    median = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    min_val = np.min(values)
    max_val = np.max(values)

    print(f"{label} Summary (N={len(values)}):")
    print(f"  Mean: {mean:.4f}, Std (sample): {std:.4f}")
    print(f"  Median: {median:.4f}")
    print(f"  Min: {min_val:.4f}, Max: {max_val:.4f}")
    print(f"  IQR: {iqr:.4f} (Q1={q1:.4f}, Q3={q3:.4f})")
    print("")

    print("  Percentiles (5% step):")
    for p in range(0, 101, 5):
        val = np.percentile(values, p)
        print(f"    {p:3d}%: {val:.4f}")
    print("")

def clean(clean_path=None):
    
    files = glob.glob(f"{clean_path}/*.json")
    layer_1_num_det = []
    layer_2_num_det = []
    
    for rank_file in tqdm(files):
        with open(rank_file, 'r') as f:
            data = json.load(f)
        # pdb.set_trace()
        for k, v in data.items():

            layer_1_num_det.append(
                v.get("L1_num", 0)  # Use get to avoid KeyError if key doesn't exist
            )
            
            layer_2_num_det.append(
                v.get("L2_num", 0)
            )

    analyze_distribution(layer_1_num_det, "Layer 1")
    analyze_distribution(layer_2_num_det, "Layer 2")
    print("="*60 + "\n")
    
    
def attack(attack_path = None):
    files = glob.glob(f"{attack_path}/*.json")
    layer_1_num_det = []
    layer_2_num_det = []

    for rank_file in tqdm(files):
        with open(rank_file, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            for i in range(200):
                layer_1_num_det.append(
                    v.get(f"iter_{i}").get("L1_num", 0)  # Use get to avoid KeyError if key doesn't exist
                )
                
                layer_2_num_det.append(
                    v.get(f"iter_{i}").get("L2_num", 0)
                )

    pdb.set_trace()
    # analyze_distribution(layer_1_num_det, "Layer 1")
    # analyze_distribution(layer_2_num_det, "Layer 2")
    print("="*60 + "\n")
    
    # 0.5 + 0.25
    # Total Layer 1 Detections: 5309
    # Total Layer 2 Detections: 8200
    # Average Layer 1 Detections per file: 53.09
    # Average Layer 2 Detections per file: 82.0
    

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Object Detection Clean Run Analysis")
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("--a", action="store_true", help="Run attack analysis")
    
    args = parser.parse_args()
    path = args.path

    if args.a:
        attack(path)
    else:
        clean(path)

