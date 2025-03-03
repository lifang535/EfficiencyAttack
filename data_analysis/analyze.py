import os
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_results_folder(base_path):
    """Analyze all model folders and their subfolders for detection statistics"""
    results = {}
    
    # Get all model folders
    model_folders = [f for f in os.listdir(base_path) if f.startswith('model_') and os.path.isdir(os.path.join(base_path, f))]
    
    for model_folder in model_folders:
        model_path = os.path.join(base_path, model_folder)
        subfolder_results = analyze_model_folder(model_path, model_folder)
        results[model_folder] = subfolder_results
    
    return results

def analyze_model_folder(model_path, model_name):
    """Analyze a single model folder and all its subfolders"""
    subfolder_results = {}
    
    # Get all subfolders
    subfolders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
    
    print(f"Analyzing {model_name} with {len(subfolders)} subfolders...")
    
    for subfolder in tqdm(subfolders):
        subfolder_path = os.path.join(model_path, subfolder)
        subfolder_data = analyze_subfolder(subfolder_path)
        subfolder_results[subfolder] = subfolder_data
    
    return subfolder_results

def analyze_subfolder(subfolder_path):
    """Analyze all JSON files in a subfolder for detection statistics"""
    # Initialize counters
    iter0_counts = []
    iter0_labels = Counter()
    max_counts = []
    max_labels = Counter()
    avg_counts = []
    all_labels = Counter()
    
    # Get all JSON files
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(subfolder_path, json_file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Iterate through each iteration in the file
            iter0_data = data.get('0', {})
            iter0_count = iter0_data.get('count', 0)
            iter0_counts.append(iter0_count)
            
            # Count labels for iteration 0
            for label in iter0_data.get('labels', []):
                iter0_labels[label] += 1
            
            # Find max count across all iterations
            max_count = 0
            max_iter = '0'
            
            for iter_key, iter_data in data.items():
                iter_count = iter_data.get('count', 0)
                if iter_count > max_count:
                    max_count = iter_count
                    max_iter = iter_key
            
            max_counts.append(max_count)
            
            # Count labels for max iteration
            for label in data.get(max_iter, {}).get('labels', []):
                max_labels[label] += 1
            
            # Calculate average count across all iterations
            total_count = 0
            iter_count = 0
            
            for iter_key, iter_data in data.items():
                count = iter_data.get('count', 0)
                total_count += count
                iter_count += 1
                
                # Count all labels across all iterations
                for label in iter_data.get('labels', []):
                    all_labels[label] += 1
            
            if iter_count > 0:
                avg_counts.append(total_count / iter_count)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Calculate statistics
    results = {
        'iter0_stats': {
            'avg_count': np.mean(iter0_counts) if iter0_counts else 0,
            'max_count': max(iter0_counts) if iter0_counts else 0,
            'min_count': min(iter0_counts) if iter0_counts else 0,
            'most_common_labels': iter0_labels.most_common(5)
        },
        'max_iter_stats': {
            'avg_count': np.mean(max_counts) if max_counts else 0,
            'max_count': max(max_counts) if max_counts else 0,
            'min_count': min(max_counts) if max_counts else 0,
            'most_common_labels': max_labels.most_common(5)
        },
        'avg_stats': {
            'avg_count': np.mean(avg_counts) if avg_counts else 0,
            'max_avg_count': max(avg_counts) if avg_counts else 0,
            'min_avg_count': min(avg_counts) if avg_counts else 0,
            'most_common_labels': all_labels.most_common(5)
        }
    }
    
    return results

def generate_summary_report(results):
    """Generate a summary report of the analysis results"""
    report = []
    
    for model_name, model_data in results.items():
        report.append(f"## {model_name} Summary")
        
        for subfolder_name, subfolder_data in model_data.items():
            report.append(f"\n### {subfolder_name}")
            
            # Iteration 0 statistics
            iter0 = subfolder_data['iter0_stats']
            report.append(f"\nIteration 0:")
            report.append(f"  Average detection count: {iter0['avg_count']:.2f}")
            report.append(f"  Max detection count: {iter0['max_count']}")
            report.append(f"  Min detection count: {iter0['min_count']}")
            report.append("  Most common labels:")
            for label, count in iter0['most_common_labels']:
                report.append(f"    - Label {label}: {count} occurrences")
            
            # Max iteration statistics
            max_iter = subfolder_data['max_iter_stats']
            report.append(f"\nMax Detections:")
            report.append(f"  Average max detection count: {max_iter['avg_count']:.2f}")
            report.append(f"  Highest max detection count: {max_iter['max_count']}")
            report.append(f"  Lowest max detection count: {max_iter['min_count']}")
            report.append("  Most common labels in max detection frames:")
            for label, count in max_iter['most_common_labels']:
                report.append(f"    - Label {label}: {count} occurrences")
            
            # Average statistics
            avg_stats = subfolder_data['avg_stats']
            report.append(f"\nAverages Across All Iterations:")
            report.append(f"  Average detection count: {avg_stats['avg_count']:.2f}")
            report.append(f"  Max average detection count: {avg_stats['max_avg_count']:.2f}")
            report.append(f"  Min average detection count: {avg_stats['min_avg_count']:.2f}")
            report.append("  Most common labels across all iterations:")
            for label, count in avg_stats['most_common_labels']:
                report.append(f"    - Label {label}: {count} occurrences")
            
            report.append("\n" + "-"*50)
    
    return "\n".join(report)

def main():
    # Set the base path to the results folder
    base_path = "../results"
    
    # Analyze all results
    results = analyze_results_folder(base_path)
    
    # Generate and print summary report
    report = generate_summary_report(results)
    print(report)
    
    # Save report to file
    with open("detection_analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"\nReport saved to detection_analysis_report.md")
    
    # Optional: Generate visualizations
    generate_visualizations(results)

def generate_visualizations(results):
    """Generate visualizations of the analysis results"""
    # Create a directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    for model_name, model_data in results.items():
        # Prepare data for plotting
        subfolder_names = []
        iter0_avgs = []
        max_avgs = []
        overall_avgs = []
        
        for subfolder_name, subfolder_data in model_data.items():
            subfolder_names.append(subfolder_name)
            iter0_avgs.append(subfolder_data['iter0_stats']['avg_count'])
            max_avgs.append(subfolder_data['max_iter_stats']['avg_count'])
            overall_avgs.append(subfolder_data['avg_stats']['avg_count'])
        
        # Create bar chart for detection counts
        plt.figure(figsize=(12, 8))
        x = np.arange(len(subfolder_names))
        width = 0.25
        
        plt.bar(x - width, iter0_avgs, width, label='Iteration 0')
        plt.bar(x, max_avgs, width, label='Max Iterations')
        plt.bar(x + width, overall_avgs, width, label='Overall Average')
        
        plt.xlabel('Subfolders')
        plt.ylabel('Average Detection Count')
        plt.title(f'{model_name} - Average Detection Counts by Subfolder')
        plt.xticks(x, subfolder_names, rotation=90)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f"visualizations/{model_name}_detection_counts.png")
        plt.close()
        
        # Create top labels visualization
        # For simplicity, we'll just use the top labels from iteration 0
        for subfolder_name, subfolder_data in model_data.items():
            top_labels = subfolder_data['iter0_stats']['most_common_labels']
            if not top_labels:
                continue
                
            labels, counts = zip(*top_labels)
            
            plt.figure(figsize=(10, 6))
            plt.bar(labels, counts)
            plt.xlabel('Label ID')
            plt.ylabel('Count')
            plt.title(f'{model_name} - {subfolder_name} - Top Labels in Iteration 0')
            plt.tight_layout()
            
            plt.savefig(f"visualizations/{model_name}_{subfolder_name}_top_labels.png")
            plt.close()

if __name__ == "__main__":
    main()