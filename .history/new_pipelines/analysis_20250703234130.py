import os
import sys
import pdb
import glob
import json
import random
import numpy as np
from tqdm import tqdm
sys.path.append("../")
import matplotlib.pyplot as plt
import utils; utils.set_all_seeds(0)


def analyze_distribution(values, label):
    """
    Analyzes the distribution of a list of numerical values and returns the
    results as a dictionary with JSON-serializable types.

    Args:
        values (list or np.ndarray): A list of numbers to analyze.
        label (str): A label for the dataset, used as a key in the output dict.

    Returns:
        dict: A dictionary containing the statistical summary.
    """
    # Ensure input is a NumPy array for efficient calculations
    values = np.array(values)
    
    # Handle empty input gracefully
    if values.size == 0:
        return {
            "label": label,
            "count": 0,
            "mean": None,
            "std_dev": None,
            "median": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "percentiles": {f"{p}%": None for p in range(0, 101, 5)}
        }

    # --- Core Statistical Calculations ---
    mean = np.mean(values)
    # ddof=1 for sample standard deviation, which is standard practice
    std = np.std(values, ddof=1) 
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Calculate percentiles and Interquartile Range (IQR)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    # --- Detailed Percentiles ---
    percentiles_dict = {}
    for p in range(0, 101, 5):
        # Convert NumPy type to native Python float here
        percentiles_dict[f"{p}%"] = float(np.percentile(values, p))

    # --- Assemble Results into a Dictionary ---
    # All NumPy numeric types are converted to native Python floats to ensure
    # they are JSON serializable.
    summary_dict = {
        "label": label,
        "count": len(values), # len() returns a native Python int
        "mean": float(mean),
        "std_dev": float(std),
        "median": float(median),
        "min": float(min_val),
        "max": float(max_val),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "percentiles": percentiles_dict # Values are already converted
    }
    
    return summary_dict

def clean(clean_path=None, output_file="analysis_results.json"):
    """
    Processes JSON files to extract detection numbers, analyzes their statistical
    distribution, and saves the results to a JSON file.

    Args:
        clean_path (str, optional): Path to the directory with input JSON files.
        output_file (str, optional): Path to save the output JSON results.
    """
    if not clean_path:
        print("Error: Please provide a path to the directory containing JSON files.")
        return

    files = glob.glob(os.path.join(clean_path, "*.json"))
    if not files:
        print(f"Warning: No JSON files found in '{clean_path}'.")
        return

    # Lists to aggregate data from all files
    layer_1_num_det = []
    layer_2_num_det = []

    # Loop through each file and extract data
    for rank_file in tqdm(files, desc="Processing files"):
        with open(rank_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file: {rank_file}. Skipping.")
                continue
        
        # Iterate through the top-level keys in the JSON data
        for k, v in data.items():
            # Ensure v is a dictionary before calling .get()
            if isinstance(v, dict):
                layer_1_num_det.append(v.get("L1_num", 0))
                layer_2_num_det.append(v.get("L2_num", 0))

    # --- Perform Analysis ---
    # Get the statistical summary for each layer
    layer_1_stats = analyze_distribution(layer_1_num_det, "Layer 1 Detections")
    layer_2_stats = analyze_distribution(layer_2_num_det, "Layer 2 Detections")
    
    # Combine all results into a final dictionary
    final_results = {
        "source_directory": os.path.abspath(clean_path),
        "file_count": len(files),
        "analysis": [
            layer_1_stats,
            layer_2_stats
        ]
    }

    # --- Save Results to JSON ---
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Write the dictionary to a JSON file with pretty printing
    with open(f"{clean_path}/{output_file}", 'w') as f:
        json.dump(final_results, f, indent=4)
        

def attack(attack_path=None):
    """
    Processes attack data from JSON files to visualize loss and detection metrics
    as line plots across iterations.

    Args:
        attack_path (str, optional): The path to the directory containing the JSON files.
                                     Defaults to the current directory if None.
    """
    # Find all JSON files in the specified directory
    files = glob.glob(f"{attack_path}/RANK_*.json")

    # Gracefully exit if no files are found
    if not files:
        print(f"Warning: No JSON files found in the specified path: {attack_path}")
        return

    # Lists to store metrics from all files. Each element will be a list of
    # values for one file across all iterations.
    all_files_l1_num = []
    all_files_l2_num = []
    all_files_l1_loss = []
    all_files_l2_loss = []
    all_files_l_total = []

    num_iterations = 200 # Define the number of iterations to process

    # Loop through each file to extract data
    for rank_file in tqdm(files, desc="Processing JSON files"):
        with open(rank_file, 'r') as f:
            data = json.load(f)

        # Assuming a single top-level key per file, as in the original structure
        data_key = next(iter(data))
        v = data[data_key]

        # Temporary lists to hold data for the current file
        current_file_l1_num = []
        current_file_l2_num = []
        current_file_l1_loss = []
        current_file_l2_loss = []
        current_file_l_total = []

        # Extract metrics for each iteration
        for i in range(num_iterations):
            iter_data = v.get(f"iter_{i}", {})  # Use .get with a default empty dict

            current_file_l1_num.append(iter_data.get("L1_num", 0))
            current_file_l2_num.append(iter_data.get("L2_num", 0))
            current_file_l1_loss.append(iter_data.get("L1_loss", 0))
            current_file_l2_loss.append(iter_data.get("L2_loss", 0))
            current_file_l_total.append(iter_data.get("L_total", 0))

        # Append the current file's data to the master lists
        all_files_l1_num.append(current_file_l1_num)
        all_files_l2_num.append(current_file_l2_num)
        all_files_l1_loss.append(current_file_l1_loss)
        all_files_l2_loss.append(current_file_l2_loss)
        all_files_l_total.append(current_file_l_total)

    # --- Data Aggregation and Calculation ---

    # Convert lists to NumPy arrays for efficient vectorized operations
    # The shape will be (number_of_files, num_iterations)
    all_files_l1_num_np = np.array(all_files_l1_num)
    all_files_l2_num_np = np.array(all_files_l2_num)
    all_files_l1_loss_np = np.array(all_files_l1_loss)
    all_files_l2_loss_np = np.array(all_files_l2_loss)
    all_files_l_total_np = np.array(all_files_l_total)

    # Calculate the mean across files for each iteration (axis=0)
    # The result is a 1D array of length num_iterations
    mean_l1_num_per_iter = np.mean(all_files_l1_num_np, axis=0)
    mean_l2_num_per_iter = np.mean(all_files_l2_num_np, axis=0)
    mean_l1_loss_per_iter = np.mean(all_files_l1_loss_np, axis=0)
    mean_l2_loss_per_iter = np.mean(all_files_l2_loss_np, axis=0)
    mean_total_loss_per_iter = np.mean(all_files_l_total_np, axis=0)

    iterations = np.arange(num_iterations)

    # --- Plotting ---

    # Create a figure and a set of subplots (2 rows, 1 column)
    # `sharex=True` makes the x-axis shared between the two plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Attack Metrics Across Iterations (Averaged)', fontsize=18, weight='bold')

    # Subplot 1: Losses
    ax1.plot(iterations, mean_total_loss_per_iter, label='Total Loss ($L_{total}$)', color='red', linewidth=2)
    ax1.plot(iterations, mean_l1_loss_per_iter, label='Layer 1 Loss ($L_1$)', linestyle='--', color='darkblue')
    ax1.plot(iterations, mean_l2_loss_per_iter, label='Layer 2 Loss ($L_2$)', linestyle='--', color='darkgreen')
    ax1.set_ylabel('Average Loss Value', fontsize=12)
    ax1.set_title('Average Loss vs. Iteration', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot 2: Number of Detections
    ax2.plot(iterations, mean_l1_num_per_iter, label='Layer 1 Detections', color='cyan', linewidth=2)
    ax2.plot(iterations, mean_l2_num_per_iter, label='Layer 2 Detections', color='magenta', linewidth=2)
    ax2.set_xlabel('Iteration Number', fontsize=12)
    ax2.set_ylabel('Average Number of Detections', fontsize=12)
    ax2.set_title('Average Detections vs. Iteration', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Improve layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.savefig(f"{attack_path}/attack_analysis_plot.png", dpi=800, bbox_inches='tight')


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

