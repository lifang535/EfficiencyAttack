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


def clean():
    clean_path = "/home/ubuntu/tingxi/EfficiencyAttack/new_pipelines/saved/clean/detr"
    rank_0_files = glob.glob(f"{clean_path}/RANK_0*")
    for rank_0_file in tqdm(rank_0_files):
        cmm = rank_0_file.split("/")[-1].replace("RANK_0", "")
        all_rank_files = glob.glob(f"{clean_path}/RANK_*{cmm}")
        for all_rank_file in all_rank_files:
            layer_1_num_det = []
            layer_2_num_det = []
            with open(all_rank_file, 'r') as f:
                data = json.load(f)
            for key, value in data.items():
                if "layer_1_num_det" in key:
                    layer_1_num_det.append(value)
                if "layer_2_num_det" in key:
                    layer_2_num_det.append(value)
                    
        mean_1 = np.mean(layer_1_num_det)
        mean_2 = np.mean(layer_2_num_det)
        std_1 = np.std(layer_1_num_det)
        std_2 = np.std(layer_2_num_det)
        print(f"File: {all_rank_file}")
        print(f"Layer 1 Mean: {mean_1}, Std: {std_1}")
        print(f"Layer 2 Mean: {mean_2}, Std: {std_2}")
        print("Layer 1 distribution:")
        print(f"1 sigma: {mean_1 - std_1}, 2 sigma: {mean_1 + std_1}")
        print(f"2 sigma: {mean_1 - 2*std_1}, 2 sigma: {mean_1 + 2*std_1}")
        print(f"3 sigma: {mean_1 - 3*std_1}, 3 sigma: {mean_1 + 3*std_1}")
        print("Layer 2 distribution:")
        print(f"1 sigma: {mean_2 - std_2}, 2 sigma: {mean_2 + std_2}")
        print(f"2 sigma: {mean_2 - 2*std_2}, 2 sigma: {mean_2 + 2*std_2}")
        print(f"3 sigma: {mean_2 - 3*std_2}, 3 sigma: {mean_2 + 3*std_2}")
        print("\n" + "="*50 + "\n")
    # Total Layer 2 Detections: 0
    # Average Layer 1 Detections per file: 4.926
    # Average Layer 2 Detections per file: 0.0
    
    
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
    clean()
    # attack(
    #     attack_path = "./saved/attack/object_detection_0.5"
    # )

