import os
import json
from collections import defaultdict

def collect_json_files(base_dir):
    # 创建一个嵌套的默认字典来存储分类的json路径
    classified_paths = defaultdict(lambda: defaultdict(list))
    total_count = 0

    # 遍历目录
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                # 获取相对路径部分进行分类
                rel_path = os.path.relpath(root, base_dir)
                path_parts = rel_path.split(os.sep)
                
                # 如果有至少两级目录
                if len(path_parts) >= 2:
                    model_group = path_parts[0]  # 例如 model_0
                    model_type = path_parts[1]   # 例如 overload_tgt_none
                    
                    # 保存文件路径
                    full_path = os.path.join(root, file)
                    classified_paths[model_group][model_type].append(full_path)
                    total_count += 1
    
    # 打印统计信息
    print(f"{'='*10} JSON文件路径统计 {'='*10}")
    
    group_totals = {}
    max_model_type_len = max([len(model_type) for group in classified_paths for model_type in classified_paths[group]], default=0)
    
    for group in sorted(classified_paths.keys()):
        group_total = 0
        print(f"\n[{group}]")
        
        for model_type in sorted(classified_paths[group].keys()):
            count = len(classified_paths[group][model_type])
            group_total += count
            print(f"=== {model_type.ljust(max_model_type_len)} json file count: {count:5d} ===")
        
        group_totals[group] = group_total
        print(f"=== {'group_total'.ljust(max_model_type_len)} json file count: {group_total:5d} ===")
    
    print(f"\n{'='*10} 总计 {'='*10}")
    for group, count in group_totals.items():
        print(f"[{group}] 总计: {count:5d} JSON文件")
    print(f"总文件数: {total_count:5d} JSON文件")
    
    return classified_paths

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际路径
    base_directory = "~/tingxi/EfficiencyAttack/results"
    # 展开用户目录（处理~符号）
    base_directory = os.path.expanduser(base_directory)
    
    json_files = collect_json_files(base_directory)
    
    # 如果你想将结果保存到文件中
    # with open("json_files_classification.json", "w") as f:
    #     json.dump({k: dict(v) for k, v in json_files.items()}, f, indent=2)