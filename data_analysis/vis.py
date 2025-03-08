from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import pdb


base_path = "../results"
base_depth = base_path.rstrip(os.sep).count(os.sep)
grouping = ["by model", "ablation", "targeted", "untargeted"]


def get_paths(base_path):
    levels = {}  # Dictionary to store directories by depth

    for root, dirs, files in os.walk(base_path):
        depth = root.count(os.sep) - base_depth  # Compute depth from base directory

        if depth not in levels:
            levels[depth] = []
        
        levels[depth].extend(os.path.join(root, d) for d in dirs)

    return levels[1]
    
    
def unit_analysis(path):
    json_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
    count_list = []
    labels_list = []
    time_list = []
    for json_file in json_list:
        with open(json_file, 'r') as f:
            data = json.load(f)
            per_count_list = []
            per_labels_list = []
            per_time_list = []
            for idx in range(len(data)):
                count, labels, time = parse_json(data[str(idx)])
                per_count_list.append(count)
                per_labels_list.append(labels)
                per_time_list.append(time)
                
            count_list.append(per_count_list)
            labels_list.append(per_labels_list)
            time_list.append(per_time_list)
            
    plot_all_count_lists(count_list, path)
    
        
def plot_all_count_lists(count_list, path):
    """
    Plot all per_count_list elements from count_list on a single graph.
    
    Args:
        count_list: List of per_count_list elements
        path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create x-axis values for each iteration
    for i, per_count_list in enumerate(count_list):
        iterations = range(len(per_count_list))
        plt.plot(iterations, per_count_list, label=f'Run {i+1}', alpha=0.7)
    
    plt.title('Count Changes Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    
    # Add legend only if there are multiple runs
    if len(count_list) > 1:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create directory for plots if it doesn't exist
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the plot
    plot_path = os.path.join(plot_dir, 'count_iterations.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def parse_json(data):
    count = data["count"]
    labels = data["labels"]
    time = data["time"]
    return (count, labels, time)


if __name__ == "__main__":
    path = "../results/model_0/teaspoon_tgt_68"
    unit_analysis(path)
    
    
    