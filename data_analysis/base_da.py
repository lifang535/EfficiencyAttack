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
import glob
import torch
from scipy.stats import kendalltau
from concurrent.futures import ThreadPoolExecutor, as_completed


base_path = "../results"
base_depth = base_path.rstrip(os.sep).count(os.sep)
grouping = ["by model", "ablation", "targeted", "untargeted"]


def get_paths(base_path, model_id=None, algorithm=None, target_idx=None):
    levels = {}  # Dictionary to store directories by depth

    for root, dirs, files in os.walk(base_path):
        depth = root.count(os.sep) - base_depth  # Compute depth from base directory

        if depth not in levels:
            levels[depth] = []
        
        levels[depth].extend(os.path.join(root, d) for d in dirs)
    
    if model_id in [0,1,2]:
        print(f"Filtering for model id: {model_id}")
        paths = []
        for p in levels[1]:
            if f"model_{model_id}" in p:
                paths.append(p)
        return paths
    elif algorithm in ["overload", "phantom", "slowtrack"]:
        print(f"Filtering for algorithm: {algorithm}")
        paths = []
        for p in levels[1]:
            if algorithm in p:
                paths.append(p)
        return paths
    elif algorithm == "tgt_none":
        print(f"Filtering for non-targeted attacks")
        paths = []
        for p in levels[1]:
            if "tgt_none" in p and "tea" in p:
                paths.append(p)
        return paths
    elif algorithm == "targeted":
        print(f"Filtering for targeted attacks on Teaspoon")
        paths = []
        for p in levels[1]:
            if "tgt_none" not in p and "tea" in p:
                paths.append(p)
        return paths
    elif target_idx == True:

        print(f"Filtering for target index: {str(target_idx)}")
        paths = []
        for p in levels[1]:
            if "none" not in p:
                paths.append(p)
        return paths
    else:
        return levels


def get_json_paths(path):
    return glob.glob(os.path.join(path, "*.json"))
    

def process_json_label(path_to_json):
    try:
        with open(path_to_json) as f:
            data = json.load(f)
        counter = torch.zeros(200, 5)
        for i in range(len(data)):
            labels = data[str(i)]["labels"]
            person = labels.count(0) 
            car = labels.count(2)
            microwave = labels.count(68)
            giraffe = labels.count(23)
            person_car = labels.count(0) + labels.count(2)
            
            counter[i][0] += person
            counter[i][1] += car
            counter[i][2] += microwave
            counter[i][3] += giraffe
            counter[i][4] += person_car
        
        return counter
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {path_to_json}: {e}")
        return torch.zeros(200, 5)  # Return empty counter for this file


def process_json(path_to_json, targeted_label_id):
    try:
        with open(path_to_json, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Warning] Error parsing JSON file '{path_to_json}': {e}")
        print("[Warning] Skipping this file and continuing...\n")
        return [] 
    
    results = []
    for i in range(len(data)):  #  200 iteration
        # iteration loss
        loss_1, loss_2, loss_3, loss_4 = data[str(i)]["loss"]
        labels = data[str(i)]["labels"]
        targeted_count = labels.count(targeted_label_id)
        results.append([loss_1, loss_2, loss_3, loss_4, targeted_count])

    return results


def baseline():
    for a in ["overload", "phantom", "slowtrack"]:
        paths = get_paths(base_path,
                        model_id=None,
                        algorithm=a,
                        target_idx=None)
        
        with open("baseline_results.txt", "a") as f:
            for p in paths:
                json_p = get_json_paths(p)[:100]
                counter = torch.zeros(200, 5)
                for jp in tqdm(json_p, desc=f"Processing JSON files: {p}"):
                    jp_counter = process_json_label(jp)
                    counter += jp_counter
                
                k = 10
                
                # table header
                f.write(f"p: {p}\n")
                f.write("-" * 50 + "\n")
                # fixed-width formatting
                f.write(f"{'Column':<8} | {'Rank':<6} | {'Position':<10} | {'Value':<12}\n")
                f.write("-" * 50 + "\n")
                
                col_names = ['person', 'car', 'microwave oven', 'giraffe', 'person + car']
                
                # top k values for each column
                for i in range(5):
                    column_data = counter[:, i]
                    top_values, top_indices = torch.topk(column_data, k)
                    
                    # write top k values to file
                    for rank, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), 1):
                        f.write(f"{col_names[i]:<8} | {rank:<6} | {idx:<10} | {val:<12.4f}\n")
                    
                    # new line
                    f.write("\n")
                
                # add space between tables
                f.write("=" * 50 + "\n\n")


def non_tgt():

    paths = get_paths(base_path,
                    model_id=None,
                    algorithm="tgt_none",
                    target_idx=None)
    
    problematic_files = []
    with open("non_tgt_results.txt", "a") as f:
        for p in paths:
            json_p = get_json_paths(p)[:100]
            counter = torch.zeros(200, 5)
            for jp in tqdm(json_p, desc="Processing JSON files"):
                try:
                    jp_counter = process_json_label(jp)
                    counter += jp_counter
                except Exception as e:
                    problematic_files.append((jp, str(e)))
                    print(f"Error processing {jp}: {e}")
                    continue
            
            k = 10
            
            # table header
            f.write(f"p: {p}\n")
            f.write("-" * 50 + "\n")
            # fixed-width formatting
            f.write(f"{'Column':<8} | {'Rank':<6} | {'Position':<10} | {'Value':<12}\n")
            f.write("-" * 50 + "\n")
            
            col_names = ['person', 'car', 'microwave oven', 'giraffe', 'person + car']
            
            # top k values for each column
            for i in range(5):
                column_data = counter[:, i]
                top_values, top_indices = torch.topk(column_data, k)
                
                # write top k values to file
                for rank, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), 1):
                    f.write(f"{col_names[i]:<8} | {rank:<6} | {idx:<10} | {val:<12.4f}\n")
                
                # new line
                f.write("\n")
            
            # add space between tables
            f.write("=" * 50 + "\n\n")
            
    if problematic_files:
        with open("problematic_files.log", "w") as error_log:
            error_log.write("Files that couldn't be processed:\n")
            for file_path, error in problematic_files:
                error_log.write(f"{file_path}: {error}\n")

def tea_tgt():

    paths = get_paths(base_path,
                    model_id=None,
                    algorithm="targeted",
                    target_idx=None)
    
    problematic_files = []
    with open("tea_tgt_results.txt", "a") as f:
        for p in paths:
            json_p = get_json_paths(p)[:100]
            counter = torch.zeros(200, 5)
            for jp in tqdm(json_p, desc="Processing JSON files"):
                try:
                    jp_counter = process_json_label(jp)
                    counter += jp_counter
                except Exception as e:
                    problematic_files.append((jp, str(e)))
                    print(f"Error processing {jp}: {e}")
                    continue
            
            k = 10
            
            # table header
            f.write(f"p: {p}\n")
            f.write("-" * 50 + "\n")
            # fixed-width formatting
            f.write(f"{'Column':<8} | {'Rank':<6} | {'Position':<10} | {'Value':<12}\n")
            f.write("-" * 50 + "\n")
            
            col_names = ['person', 'car', 'microwave oven', 'giraffe', 'person + car']
            
            # top k values for each column
            for i in range(5):
                column_data = counter[:, i]
                top_values, top_indices = torch.topk(column_data, k)
                
                # write top k values to file
                for rank, (val, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist()), 1):
                    f.write(f"{col_names[i]:<8} | {rank:<6} | {idx:<10} | {val:<12.4f}\n")
                
                # new line
                f.write("\n")
            
            # add space between tables
            f.write("=" * 50 + "\n\n")
            
    if problematic_files:
        with open("problematic_files.log", "w") as error_log:
            error_log.write("Files that couldn't be processed:\n")
            for file_path, error in problematic_files:
                error_log.write(f"{file_path}: {error}\n")
                
def loss(path_to_jsons):
    
    if "0" in path_to_jsons:
        target_idx = 0
    if "2" in path_to_jsons:
        target_idx = 2
    if "68" in path_to_jsons:
        target_idx = 68
    if "23" in path_to_jsons:
        target_idx = 23
    if "0_2" in path_to_jsons:
        target_idx = [0,2]
    else:
        pass
    
    jsons_path = get_json_paths(path_to_jsons)
    print(f"Found {len(jsons_path)} JSON files in {path_to_jsons}")
    all_data = []
    
    for j in tqdm(jsons_path, desc="Processing JSON files"):
        iteration_data = process_json(j, 0)
        all_data.extend(iteration_data)
    
    columns = ["loss_1", "loss_2", "loss_3", "loss_4", "targeted_count"]
    df = pd.DataFrame(all_data, columns=columns)
    
    # ========== Log transform the targeted_count ==========
    df["targeted_count_log"] = np.log1p(df["targeted_count"])  # log( x + 1 )
    # ========== (A) Pearson ==========
    pearson_corr_matrix = df.corr(method="pearson")
    pearson_corr_loss_targeted = df[["loss_1","loss_2","loss_3","loss_4"]].corrwith(df["targeted_count"])
    pearson_corr_loss_targeted_log = df[["loss_1","loss_2","loss_3","loss_4"]].corrwith(df["targeted_count_log"])
    
    # ========== (B) Spearman ==========
    spearman_corr_matrix = df.corr(method="spearman")
    spearman_corr_loss_targeted = df[["loss_1","loss_2","loss_3","loss_4"]].corrwith(df["targeted_count"], method="spearman")
    
    # ========== (C) Kendall’s τ for one pair as an example ==========
    tau, p_val = kendalltau(df["loss_1"], df["targeted_count"])
    
    # ========== Prepare the output text ==========
    output_lines = []

    output_lines.append("========= (A) Pearson ==========\n")
    output_lines.append("Pearson Correlation Matrix:\n")
    output_lines.append(str(pearson_corr_matrix))
    output_lines.append("\n")
    
    output_lines.append("\nPearson correlation (loss_i vs. targeted_count):\n")
    output_lines.append(str(pearson_corr_loss_targeted))
    output_lines.append("\n")
    
    output_lines.append("\nPearson correlation (loss_i vs. log(targeted_count+1)):\n")
    output_lines.append(str(pearson_corr_loss_targeted_log))
    output_lines.append("\n")

    output_lines.append("\n========= (B) Spearman ==========\n")
    output_lines.append("Spearman Correlation Matrix:\n")
    output_lines.append(str(spearman_corr_matrix))
    output_lines.append("\n")

    output_lines.append("\nSpearman correlation (loss_i vs. targeted_count):\n")
    output_lines.append(str(spearman_corr_loss_targeted))
    output_lines.append("\n")

    output_lines.append("\n========= (C) Kendall’s τ ==========\n")
    output_lines.append(f"Kendall’s τ for loss_1 vs. targeted_count: {tau}, p-value={p_val}\n")

    # ========== Write to file ==========
    name = os.path.join("./loss", os.path.basename(os.path.dirname(path_to_jsons)), os.path.basename(path_to_jsons) + ".txt")
    os.makedirs(os.path.join("./loss", os.path.basename(os.path.dirname(path_to_jsons))), exist_ok=True)
    with open(name, 'w', encoding='utf-8') as f:
        f.write("".join(output_lines))

    # Optional: You can still print to console if you want
    # print("".join(output_lines))

def vis(path_list):
    """Visualizes the loss terms and targeted count for a single image over 200 iterations."""

    for path_to_json in get_json_paths(path_list):
        # Extract target index from the filename
        if "0" in path_to_json:
            target_idx = 0
        elif "2" in path_to_json:
            target_idx = 2
        elif "68" in path_to_json:
            target_idx = 68
        elif "23" in path_to_json:
            target_idx = 23
        elif "0_2" in path_to_json:
            target_idx = [0,2]
        else:
            target_idx = 0  # Default to 0 if no target index found
        # Extract loss data for this single image
        iteration_data = process_json(path_to_json, target_idx)

        # Convert to DataFrame
        columns = ["loss_1", "loss_2", "loss_3", "loss_4", "targeted_count"]
        df = pd.DataFrame(iteration_data, columns=columns)

        # Ensure exactly 200 iterations
        df = df.iloc[:200]

        # X-axis (Iterations)
        iterations = np.arange(1, 201)

        # Create a figure
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot loss values (left Y-axis)
        ax1.plot(iterations, df["loss_1"], label="Loss 1", color="blue", linestyle="-")
        ax1.plot(iterations, df["loss_2"], label="Loss 2", color="red", linestyle="--")
        ax1.plot(iterations, df["loss_3"], label="Loss 3", color="green", linestyle="-.")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss Values")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Create secondary y-axis for targeted count
        ax2 = ax1.twinx()
        ax2.plot(iterations, df["targeted_count"], label="Targeted Count", color="black", linestyle="dotted")
        ax2.set_ylabel("Targeted Count")
        ax2.legend(loc="upper left")

        plt.title(f"Loss Terms and Targeted Count Over 200 Iterations\n{os.path.basename(path_to_json)}")

        # Save the visualization per image
        output_dir = os.path.join("./loss_plt", os.path.basename(os.path.dirname(path_to_json)))
        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, f"{os.path.basename(path_to_json)}.png")
        plt.savefig(vis_path)
        plt.close()
        print(f"Saved loss visualization to {vis_path}")

if __name__ == "__main__":
    # non_tgt()
    baseline()
    tea_tgt()
    paths = get_paths(base_path,
                    model_id=None,
                    algorithm=None,
                    target_idx=True)
    print("Processing : ", len(paths))
    for p in paths:
        loss(p)
        
        
    pass

            
            