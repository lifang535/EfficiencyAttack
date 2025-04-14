import json
import glob
import numpy as np
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

base_dirs = [f"../ablation/results/model_{str(i)}" for i in [0,1,2]]
sub_dirs = ["norm_tgt_0", "area_tgt_0", "norm_and_area_tgt_0"]
# sub_dirs = ["norm_and_area_tgt_0"]

def get_dirs(base_dirs, sub_dirs):
    dirs = []
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            dirs.append(f"{base_dir}/{sub_dir}")
    return dirs

def json_helper(folder_dir):
    """process json files within a folder"""
    jsons = glob.glob(f"{folder_dir}/*.json")
    
    # Data containers for this folder
    total_losses = []
    cls_losses = []
    person_counts = []
    area_losses = []
    norm_losses = []
    
    for j_dir in jsons:
        with open(j_dir) as f:
            data = json.load(f)
            for i in range(len(data)):
                total_loss = data[str(i)]['loss']
                cls_loss = data[str(i)]["cls loss"]
                person_count = data[str(i)]["labels"].count(0)
                
                # Append data for each sample
                total_losses.append(total_loss)
                cls_losses.append(cls_loss)
                person_counts.append(person_count)
                
                # Handle optional losses
                if 'area loss' in data[str(i)]:
                    area_losses.append(data[str(i)]['area loss'])
                
                if 'norm loss' in data[str(i)]:
                    norm_losses.append(data[str(i)]['norm loss'])
    
    # Compile results
    results = {
        'folder': folder_dir,
        'total_losses': total_losses,
        'cls_losses': cls_losses,
        'person_counts': person_counts,
        'area_losses': area_losses if area_losses else None,
        'norm_losses': norm_losses if norm_losses else None
    }
    
    return results

def calculate_correlations(folder_results):
    """Calculate correlations between different losses and person count"""
    
    correlations = {}
    folder = folder_results['folder']
    correlations['folder'] = folder
    
    # Calculate correlation for total loss vs person count
    if folder_results['total_losses'] and folder_results['person_counts']:
        corr, p_value = spearmanr(folder_results['total_losses'], folder_results['person_counts'])
        correlations['total_loss_vs_person_count'] = {
            'correlation': corr,
            'p_value': p_value
        }
    
    # Calculate correlation for cls loss vs person count
    if folder_results['cls_losses'] and folder_results['person_counts']:
        corr, p_value = spearmanr(folder_results['cls_losses'], folder_results['person_counts'])
        correlations['cls_loss_vs_person_count'] = {
            'correlation': corr,
            'p_value': p_value
        }
    
    # Calculate correlation for area loss vs person count (if available)
    if folder_results['area_losses'] and folder_results['person_counts']:
        corr, p_value = spearmanr(folder_results['area_losses'], folder_results['person_counts'])
        correlations['area_loss_vs_person_count'] = {
            'correlation': corr,
            'p_value': p_value
        }
    
    # Calculate correlation for norm loss vs person count (if available)
    if folder_results['norm_losses'] and folder_results['person_counts']:
        corr, p_value = spearmanr(folder_results['norm_losses'], folder_results['person_counts'])
        correlations['norm_loss_vs_person_count'] = {
            'correlation': corr,
            'p_value': p_value
        }
    
    return correlations

def plot_correlation(folder_results, output_dir="correlation_plots"):
    """Generate scatter plots to visualize correlations"""
    folder_name = os.path.basename(folder_results['folder'])
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot total loss vs person count
    if folder_results['total_losses'] and folder_results['person_counts']:
        plt.figure(figsize=(10, 6))
        plt.scatter(folder_results['person_counts'], folder_results['total_losses'], alpha=0.5)
        plt.title(f'Total Loss vs Person Count - {folder_name}')
        plt.xlabel('Person Count')
        plt.ylabel('Total Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient to plot
        corr, _ = spearmanr(folder_results['person_counts'], folder_results['total_losses'])
        plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.savefig(f"{output_dir}/{folder_name}_total_loss_correlation.png")
        plt.close()
    
    # Plot cls loss vs person count
    if folder_results['cls_losses'] and folder_results['person_counts']:
        plt.figure(figsize=(10, 6))
        plt.scatter(folder_results['person_counts'], folder_results['cls_losses'], alpha=0.5)
        plt.title(f'Classification Loss vs Person Count - {folder_name}')
        plt.xlabel('Person Count')
        plt.ylabel('Classification Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient to plot
        corr, _ = spearmanr(folder_results['person_counts'], folder_results['cls_losses'])
        plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.savefig(f"{output_dir}/{folder_name}_cls_loss_correlation.png")
        plt.close()
    
    # Plot area loss vs person count (if available)
    if folder_results['area_losses'] and folder_results['person_counts']:
        plt.figure(figsize=(10, 6))
        plt.scatter(folder_results['person_counts'], folder_results['area_losses'], alpha=0.5)
        plt.title(f'Area Loss vs Person Count - {folder_name}')
        plt.xlabel('Person Count')
        plt.ylabel('Area Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient to plot
        corr, _ = spearmanr(folder_results['person_counts'], folder_results['area_losses'])
        plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.savefig(f"{output_dir}/{folder_name}_area_loss_correlation.png")
        plt.close()
    
    # Plot norm loss vs person count (if available)
    if folder_results['norm_losses'] and folder_results['person_counts']:
        plt.figure(figsize=(10, 6))
        plt.scatter(folder_results['person_counts'], folder_results['norm_losses'], alpha=0.5)
        plt.title(f'Norm Loss vs Person Count - {folder_name}')
        plt.xlabel('Person Count')
        plt.ylabel('Norm Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add correlation coefficient to plot
        corr, _ = spearmanr(folder_results['person_counts'], folder_results['norm_losses'])
        plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.savefig(f"{output_dir}/{folder_name}_norm_loss_correlation.png")
        plt.close()

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
        
        # Generate plots
        plot_correlation(folder_results)
    
    # Print summary of correlations
    print("\nCorrelation Summary:")
    for corr in all_correlations:
        print(f"\nFolder: {corr['folder']}")
        for key, value in corr.items():
            if key != 'folder':
                print(f"  {key}: r={value['correlation']:.3f}, p={value['p_value']:.5f}")
    
    # Save correlations to JSON
    with open('correlation_results.json', 'w') as f:
        json.dump(all_correlations, f, indent=4)
    
    print("\nAnalysis complete. Results saved to correlation_results.json")

if __name__ == "__main__":
    main()