import json
import glob
import numpy as np
import os
import re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict

base_dirs = [f"../ablation/results/model_{str(i)}" for i in [0,1,2]]
sub_dirs = ["eps_2_tgt_0", "eps_8_tgt_0", "teaspoon_tgt_0"]

def get_dirs(base_dirs, sub_dirs):
    dirs = []
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            dirs.append(f"{base_dir}/{sub_dir}")
    return dirs

def json_helper(folder_dir):
    """Process JSON files within a folder
    
    Each folder contains multiple JSON files.
    Each JSON file contains data for multiple iterations.
    For each iteration, we count the number of 0s in the labels.
    """
    jsons = glob.glob(f"{folder_dir}/*.json")
    
    # Collect person counts for each iteration across all files
    # We'll use a dictionary to combine data from all files
    iteration_counts = defaultdict(int)
    iteration_losses = defaultdict(float)
    total_files_processed = 0
    
    for j_dir in jsons:
        with open(j_dir) as f:
            data = json.load(f)
            
            # Process each iteration in this file
            for i in range(len(data)):
                try:
                    # Try with string keys first (as in the updated format)
                    item_key = str(i)
                    total_loss = data[item_key]['loss']
                    person_count = data[item_key]["labels"].count(0)
                except KeyError:
                    # Fall back to integer keys if string keys don't work
                    item_key = i
                    total_loss = data[item_key]['loss']
                    person_count = data[item_key]["labels"].count(0)
                
                # Accumulate counts and losses for this iteration across all files
                iteration_counts[i] += person_count
                iteration_losses[i] += total_loss
        
        total_files_processed += 1
    
    # Convert dictionaries to sorted lists
    iterations = sorted(iteration_counts.keys())
    person_counts = [iteration_counts[i] for i in iterations]
    total_losses = [iteration_losses[i] for i in iterations]
    
    # Compile results
    results = {
        'folder': folder_dir,
        'total_losses': total_losses,
        'person_counts': person_counts,
        'iterations': iterations,
        'files_processed': total_files_processed
    }
    
    return results
            
    # Compile results
    results = {
        'folder': folder_dir,
        'total_losses': total_losses,
        'person_counts': person_counts,
        'iterations': iterations,
        'file_ids': file_ids,
    }
    
    return results

def calculate_correlations(folder_results):
    """Calculate correlations between total loss and person count"""
    
    correlations = {}
    folder = folder_results['folder']
    correlations['folder'] = folder
    
    # Calculate correlation for total loss vs person count
    if folder_results['total_losses'] and folder_results['person_counts']:
        corr, p_value = pearsonr(folder_results['total_losses'], folder_results['person_counts'])
        correlations['total_loss_vs_person_count'] = {
            'correlation': corr,
            'p_value': p_value,
            'avg_loss': np.mean(folder_results['total_losses']),
            'std_loss': np.std(folder_results['total_losses']),
            'sample_count': len(folder_results['total_losses'])
        }
    
    return correlations

def plot_correlation_by_epsilon(all_results, output_dir="ablation_plots"):
    """Generate plots to compare different epsilon values across models"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Organize results by model and epsilon
    model_eps_results = defaultdict(dict)
    
    for result in all_results:
        folder = result['folder']
        # Extract model number and epsilon from folder path
        parts = folder.split('/')
        model_name = parts[-2]  # model_0, model_1, model_2
        eps_name = parts[-1]    # eps_2_tgt_0, eps_8_tgt_0, teaspoon_tgt_0
        
        # Store the results
        model_eps_results[model_name][eps_name] = result
    
    # Plot 1: Correlation coefficients by epsilon for each model
    plt.figure(figsize=(12, 8))
    
    models = sorted(model_eps_results.keys())
    epsilon_values = ['eps_2_tgt_0', 'eps_8_tgt_0', 'teaspoon_tgt_0']
    epsilon_labels = ['ε=2', 'ε=8', 'ε=4 (Teaspoon)']
    
    # For bar plotting
    width = 0.25
    x = np.arange(len(epsilon_values))
    
    for i, model in enumerate(models):
        correlations = []
        for eps in epsilon_values:
            if eps in model_eps_results[model]:
                folder_result = model_eps_results[model][eps]
                corr, _ = pearsonr(folder_result['total_losses'], folder_result['person_counts'])
                correlations.append(corr)
            else:
                correlations.append(0)  # Default if data missing
        
        # Plot bars for this model
        plt.bar(x + i*width, correlations, width, label=model)
    
    plt.xlabel('Epsilon Value')
    plt.ylabel('Correlation Coefficient (Loss vs Person Count)')
    plt.title('Effect of Epsilon on Loss vs Person Count Correlation')
    plt.xticks(x + width, epsilon_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/epsilon_correlation_comparison.png")
    plt.close()
    
    # Plot 2: Average loss by epsilon for each model
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(models):
        avg_losses = []
        for eps in epsilon_values:
            if eps in model_eps_results[model]:
                folder_result = model_eps_results[model][eps]
                avg_loss = np.mean(folder_result['total_losses'])
                avg_losses.append(avg_loss)
            else:
                avg_losses.append(0)  # Default if data missing
        
        # Plot bars for this model
        plt.bar(x + i*width, avg_losses, width, label=model)
    
    plt.xlabel('Epsilon Value')
    plt.ylabel('Average Loss')
    plt.title('Effect of Epsilon on Average Loss')
    plt.xticks(x + width, epsilon_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/epsilon_avg_loss_comparison.png")
    plt.close()
    
    # Plot 3: Scatter plots for each model-epsilon combination
    for model in models:
        plt.figure(figsize=(15, 5))
        plot_idx = 1
        
        for eps in epsilon_values:
            if eps in model_eps_results[model]:
                folder_result = model_eps_results[model][eps]
                
                plt.subplot(1, 3, plot_idx)
                plt.scatter(folder_result['person_counts'], folder_result['total_losses'], alpha=0.5)
                
                # Add correlation coefficient to plot
                corr, _ = pearsonr(folder_result['person_counts'], folder_result['total_losses'])
                
                epsilon_label = epsilon_labels[epsilon_values.index(eps)]
                plt.title(f'{model} - {epsilon_label}\nCorr: {corr:.3f}')
                plt.xlabel('Person Count')
                plt.ylabel('Total Loss')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_epsilon_scatter_plots.png")
        plt.close()

def plot_count_vs_iteration(all_results, output_dir="ablation_plots"):
    """
    Generate plots to show person count vs. iteration for each model with different epsilon values
    Each model gets one plot with three curves (one per epsilon value)
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Organize results by model and epsilon
    model_eps_results = defaultdict(dict)
    
    for result in all_results:
        folder = result['folder']
        # Extract model number and epsilon from folder path
        parts = folder.split('/')
        model_name = parts[-2]  # model_0, model_1, model_2
        eps_name = parts[-1]    # eps_2_tgt_0, eps_8_tgt_0, teaspoon_tgt_0
        
        # Store the results
        model_eps_results[model_name][eps_name] = result
    
    models = sorted(model_eps_results.keys())
    epsilon_values = ['eps_2_tgt_0', 'eps_8_tgt_0', 'teaspoon_tgt_0']
    epsilon_labels = ['ε=2', 'ε=8', 'ε=4 (Teaspoon)']
    
    # Plot count vs. iteration for each model - one plot per model
    for model in models:
        plt.figure(figsize=(12, 8))
        
        for i, eps in enumerate(epsilon_values):
            if eps in model_eps_results[model]:
                folder_result = model_eps_results[model][eps]
                
                # Get iterations and counts
                iterations = folder_result['iterations']
                counts = folder_result['person_counts']
                
                # Plot line for this epsilon value
                epsilon_label = epsilon_labels[i]
                plt.plot(iterations, counts, marker='o', linestyle='-', label=f"{epsilon_label} ({folder_result['files_processed']} files)")
        
        plt.xlabel('Iteration')
        plt.ylabel('Person Count')
        plt.title(f'{model}: Count vs Iteration for Different Epsilon Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_count_vs_iteration.png")
        plt.close()
        
        # Also save the data to a CSV file for reference
        csv_file = f"{output_dir}/{model}_count_data.csv"
        with open(csv_file, 'w') as f:
            # Write header
            f.write("Iteration,")
            for eps_idx, eps in enumerate(epsilon_values):
                if eps in model_eps_results[model]:
                    f.write(f"{epsilon_labels[eps_idx]},")
            f.write("\n")
            
            # Write data rows
            max_iterations = max([max(model_eps_results[model][eps]['iterations']) 
                                for eps in epsilon_values if eps in model_eps_results[model]])
            
            for iter_num in range(max_iterations + 1):
                f.write(f"{iter_num},")
                for eps_idx, eps in enumerate(epsilon_values):
                    if eps in model_eps_results[model]:
                        result_data = model_eps_results[model][eps]
                        iter_index = result_data['iterations'].index(iter_num) if iter_num in result_data['iterations'] else -1
                        if iter_index >= 0:
                            f.write(f"{result_data['person_counts'][iter_index]},")
                        else:
                            f.write(",")  # Empty if no data for this iteration
                f.write("\n")

def generate_ablation_table(all_correlations):
    """Generate a table summarizing the ablation study results"""
    
    # Organize results by model and epsilon
    table_data = defaultdict(dict)
    
    for corr_data in all_correlations:
        folder = corr_data['folder']
        # Extract model number and epsilon from folder path
        parts = folder.split('/')
        model_name = parts[-2]  # model_0, model_1, model_2
        eps_name = parts[-1]    # eps_2_tgt_0, eps_8_tgt_0, teaspoon_tgt_0
        
        # Get correlation data
        if 'total_loss_vs_person_count' in corr_data:
            corr_value = corr_data['total_loss_vs_person_count']['correlation']
            p_value = corr_data['total_loss_vs_person_count']['p_value']
            avg_loss = corr_data['total_loss_vs_person_count']['avg_loss']
            std_loss = corr_data['total_loss_vs_person_count']['std_loss']
            
            table_data[model_name][eps_name] = {
                'correlation': corr_value,
                'p_value': p_value,
                'avg_loss': avg_loss,
                'std_loss': std_loss
            }
    
    # Define epsilon mapping for better readability
    eps_mapping = {
        'eps_2_tgt_0': 'ε=2',
        'eps_8_tgt_0': 'ε=8',
        'teaspoon_tgt_0': 'ε=4 (Teaspoon)'
    }
    
    # Generate markdown table
    markdown_table = "# Ablation Study Results\n\n"
    markdown_table += "## Correlation between Total Loss and Person Count\n\n"
    
    markdown_table += "| Model | Epsilon | Correlation | p-value | Avg Loss | Std Dev |\n"
    markdown_table += "|-------|---------|-------------|---------|----------|--------|\n"
    
    for model in sorted(table_data.keys()):
        for eps in ['eps_2_tgt_0', 'eps_8_tgt_0', 'teaspoon_tgt_0']:
            if eps in table_data[model]:
                data = table_data[model][eps]
                markdown_table += f"| {model} | {eps_mapping[eps]} | {data['correlation']:.4f} | {data['p_value']:.4f} | {data['avg_loss']:.4f} | {data['std_loss']:.4f} |\n"
    
    # Write table to file
    with open('ablation_study_results.md', 'w') as f:
        f.write(markdown_table)
    
    return markdown_table

def main():
    # Get all directory paths
    all_dirs = get_dirs(base_dirs, sub_dirs)
    
    # Store results for all folders
    all_results = []
    all_correlations = []
    
    # Process each folder
    for folder_dir in all_dirs:
        print(f"Processing {folder_dir}...")
        folder_results = json_helper(folder_dir)
        all_results.append(folder_results)
        
        # Calculate correlations
        correlations = calculate_correlations(folder_results)
        all_correlations.append(correlations)
    
    # Generate comparative plots
    plot_correlation_by_epsilon(all_results)
    
    # Generate count vs. iteration plots (new function)
    plot_count_vs_iteration(all_results)
    
    # Generate ablation study table
    table = generate_ablation_table(all_correlations)
    print("\nAblation Study Summary:")
    print(table)
    
    print("\nAnalysis complete. Results saved to ablation_study_results.md and plots in ablation_plots/")

if __name__ == "__main__":
    main()