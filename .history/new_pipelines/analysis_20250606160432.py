import os
import sys
import pdb
import json
import random
sys.path.append("../")
import utils; utils.set_all_seeds(0)


def clean():
    clean_path = "/home/ubuntu/tingxi/EfficiencyAttack/new_pipelines/saved/clean/detr"
    json_files = [f for f in os.listdir(clean_path) if f.endswith('.json')]
    
    total_layer_1_num_detections = 0
    total_layer_2_num_detections = 0
    file_count = len(json_files)

    pdb.set_trace()
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

