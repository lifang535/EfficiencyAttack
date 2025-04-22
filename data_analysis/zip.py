import os
import random
import zipfile

# Set random seed
random.seed(0)

# Set your target directory
# directory = "/path/to/your/directory"
dirs = [
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/overload_tgt_none",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/phantom_tgt_none",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/slowtrack_tgt_none",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/teaspoon_tgt_0",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/teaspoon_tgt_0_2",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/teaspoon_tgt_2",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/teaspoon_tgt_23",
    "/home/ubuntu/tingxi/EfficiencyAttack/saved/model_0/teaspoon_tgt_68",
]

dirs = [
    "/home/ubuntu/tingxi/saved/model_0/teaspoon_tgt_20",
    "/home/ubuntu/tingxi/saved/model_0/teaspoon_tgt_72",
    "/home/ubuntu/tingxi/saved/model_0/teaspoon_tgt_20_23",
]
def zipping(directory):
    folder_name = os.path.basename(os.path.abspath(directory))
    zip_filename = f"{folder_name}.zip"

    # Collect all .pt files
    pt_files = [f for f in os.listdir(directory) if f.endswith(".pt")]

    # Sample 100
    sampled_files = random.sample(pt_files, 100)

    # Create zip archive
    zip_path = os.path.join("./", zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in sampled_files:
            file_path = os.path.join(directory, filename)
            zipf.write(file_path, arcname=filename)  # Save without full path

    print(f"Zipped {len(sampled_files)} files to {zip_path}")
    
if __name__ == "__main__":
    for d in dirs:
        zipping(d)
