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
    std = np.std(values, ddof=1)  # 样本标准差
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

    print("  Empirical intervals:")
    print(f"    1σ range: [{mean - std:.4f}, {mean + std:.4f}]")
    print(f"    2σ range: [{mean - 2*std:.4f}, {mean + 2*std:.4f}]")
    print(f"    3σ range: [{mean - 3*std:.4f}, {mean + 3*std:.4f}]")
    print(f"    IQR range: [{q1 - 1.5*iqr:.4f}, {q3 + 1.5*iqr:.4f}]")
    print("")

def clean(clean_path=None):
    
    files = glob.glob(f"{clean_path}/*.json")
    pdb.set_trace()
    for rank_file in tqdm(files):

        layer_1_num_det = []
        layer_2_num_det = []

        with open(rank_file, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            if "layer_1_num_det" in k:
                layer_1_num_det.append(v)
            elif "layer_2_num_det" in k:
                layer_2_num_det.append(v)

        analyze_distribution(layer_1_num_det, "Layer 1")
        analyze_distribution(layer_2_num_det, "Layer 2")
        print("="*60 + "\n")
    
    
def attack(attack_path = "./saved/attack/object_detection"):
    json_files = [f for f in os.listdir(attack_path) if f.endswith('.json')]
    total_layer_1_num_detections = 0
    total_layer_2_num_detections = 0
    file_count = len(json_files)
    for json_file in json_files:
        file_path = os.path.join(attack_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            total_layer_1_num_detections += data['iteration_199']["layer_1_num_dets"] 
            total_layer_2_num_detections += data['iteration_199']["layer_2_num_dets"]
            
    print(f"Total Layer 1 Detections: {total_layer_1_num_detections}")
    print(f"Total Layer 2 Detections: {total_layer_2_num_detections}")
    print(f"Average Layer 1 Detections per file: {total_layer_1_num_detections / file_count if file_count > 0 else 0}")
    print(f"Average Layer 2 Detections per file: {total_layer_2_num_detections / file_count if file_count > 0 else 0}")
    
    
    # 0.5 + 0.25
    # Total Layer 1 Detections: 5309
    # Total Layer 2 Detections: 8200
    # Average Layer 1 Detections per file: 53.09
    # Average Layer 2 Detections per file: 82.0
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Object Detection Clean Run Analysis")
    parser.add_argument("clean_path", type=str, default=None)
    args = parser.parse_args()
    clean_path = args.clean_path
    
    clean(clean_path)
    # attack(
    #     attack_path = "./saved/attacked/detr"
    # )

