import json
import glob
import numpy as np
import os
import re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from collections import defaultdict

base_dirs = [f"../ablation/results/model_{str(i)}" for i in [0,1,2]]
sub_dirs = ["tea_100_tgt_0", "teaspoon_tgt_0", "tea_400_tgt_0"]  # teaspoon_tgt_0 代表迭代200

def get_dirs(base_dirs, sub_dirs):
    dirs = []
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            dirs.append(f"{base_dir}/{sub_dir}")
    return dirs

def json_helper(folder_dir):
    """处理文件夹中的JSON文件
    
    每个文件夹包含100个JSON文件。
    每个JSON文件包含200个迭代的数据。
    对于每个迭代，我们计算标签中0的数量。
    """
    jsons = glob.glob(f"{folder_dir}/*.json")
    print(f"Found {len(jsons)} JSON files in {folder_dir}")
    
    # 为每个迭代收集person counts
    iteration_counts = defaultdict(int)
    total_files_processed = 0
    
    for j_dir in jsons:
        try:
            with open(j_dir) as f:
                data = json.load(f)
                
                # 处理此文件中的每个迭代
                for i in range(len(data)):
                    try:
                        # 尝试先使用字符串键（更新后的格式）
                        item_key = str(i)
                        person_count = data[item_key]["labels"].count(0)
                    except (KeyError, TypeError):
                        try:
                            # 如果字符串键不能用，回退到整数键
                            item_key = i
                            person_count = data[item_key]["labels"].count(0)
                        except (KeyError, IndexError, TypeError) as e:
                            continue
                    
                    # 累计每个迭代的counts
                    iteration_counts[i] += person_count
            
            total_files_processed += 1
        except Exception as e:
            print(f"Error processing file {j_dir}: {e}")
            continue
    
    # 将字典转换为排序列表
    iterations = sorted(iteration_counts.keys())
    person_counts = [iteration_counts[i] for i in iterations]
    
    # 编译结果
    results = {
        'folder': folder_dir,
        'person_counts': person_counts,
        'iterations': iterations,
        'files_processed': total_files_processed,
        'setting': os.path.basename(folder_dir)  # tea_100_tgt_0, teaspoon_tgt_0, tea_400_tgt_0
    }
    
    if len(iterations) > 0:
        print(f"Processed {folder_dir}: {total_files_processed} files, {len(iterations)} iterations")
        print(f"  Iteration range: {min(iterations)} to {max(iterations)}")
        print(f"  Person count range: {min(person_counts)} to {max(person_counts)}")
    else:
        print(f"Warning: No valid iterations found in {folder_dir}")
    
    return results

def plot_count_vs_iteration(all_results, output_dir="ablation_plots"):
    """
    为每个模型生成count vs. iteration图，每张图显示该模型的三种迭代设置曲线
    """
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 按模型和设置组织结果
    model_iteration_results = defaultdict(dict)
    
    for result in all_results:
        folder = result['folder']
        # 从文件夹路径提取模型号和设置
        parts = folder.split('/')
        model_name = parts[-2]  # model_0, model_1, model_2
        setting_name = parts[-1]  # tea_100_tgt_0, teaspoon_tgt_0, tea_400_tgt_0
        
        # 存储结果
        model_iteration_results[model_name][setting_name] = result
    
    models = sorted(model_iteration_results.keys())
    iteration_settings = ['tea_100_tgt_0', 'teaspoon_tgt_0', 'tea_400_tgt_0']
    iteration_labels = ['Iteration 100', 'Iteration 200 (Teaspoon)', 'Iteration 400']
    line_styles = ['-', '--', '-.']  # 实线, 虚线, 点划线
    
    # 为每个模型绘制一张图，图上有三条线代表不同迭代次数
    for model in models:
        plt.figure(figsize=(12, 8))
        
        for i, setting in enumerate(iteration_settings):
            if setting in model_iteration_results[model]:
                folder_result = model_iteration_results[model][setting]
                
                # 如果没有数据，跳过
                if not folder_result['iterations'] or not folder_result['person_counts']:
                    print(f"Warning: No data for {model}/{setting}, skipping")
                    continue
                
                # 获取迭代和计数数据
                iterations = folder_result['iterations']
                counts = folder_result['person_counts']
                
                # 绘制该迭代设置的曲线
                setting_label = iteration_labels[i]
                line_style = line_styles[i % len(line_styles)]
                plt.plot(iterations, counts, linestyle=line_style, marker='o', 
                         linewidth=2, markersize=6, 
                         label=f"{setting_label} ({folder_result['files_processed']} files)")
        
        plt.xlabel('Iteration Number', fontsize=14)
        plt.ylabel('Number of Workload', fontsize=14)
        plt.title(f'{model}: Number of Workload vs Iteration Number', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_iteration_comparison.png", dpi=300)
        print(f"Saved plot for {model} to {output_dir}/{model}_iteration_comparison.png")
        plt.close()

def main():
    # 获取所有目录路径
    all_dirs = get_dirs(base_dirs, sub_dirs)
    print(f"Processing {len(all_dirs)} directories...")
    
    # 存储所有文件夹的结果
    all_results = []
    
    # 处理每个文件夹
    for folder_dir in all_dirs:
        print(f"\nProcessing {folder_dir}...")
        folder_results = json_helper(folder_dir)
        all_results.append(folder_results)
    
    # 生成count vs. iteration图
    plot_count_vs_iteration(all_results)
    
    print("\nAnalysis complete. Plots saved in ablation_plots/ directory")

if __name__ == "__main__":
    main()