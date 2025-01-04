#%%
import matplotlib.pyplot as plt
import json
# Sample data: Replace this with the actual JSON data from your files
file1_path = "/home/lxt230026/EfficiencyAttack/detr-prediction/20250102_1337_100_slow.json"
file2_path = "/home/lxt230026/EfficiencyAttack/detr-prediction/20250101_1707_100_ada.json"

# Load the data from the JSON files
with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
    file1_data = json.load(file1)
    file2_data = json.load(file2)
    
# Extracting keys (images) and values for comparison
images = list(file1_data.keys())
clean_file1 = [file1_data[img]["clean_bbox_num"] for img in images]
corrupted_file1 = [file1_data[img]["corrupted_bbox_num"] for img in images]
clean_file2 = [file2_data[img]["clean_bbox_num"] for img in images]
corrupted_file2 = [file2_data[img]["corrupted_bbox_num"] for img in images]

# Plotting the comparison
plt.figure(figsize=(12, 6))

# Clean bounding boxes comparison
plt.subplot(1, 2, 1)
plt.bar(images, clean_file1, alpha=0.6, label='File 1 - Clean', width=0.4, align='center')
plt.bar(images, clean_file2, alpha=0.6, label='File 2 - Clean', width=0.4, align='edge')
plt.title('Clean Bounding Boxes Comparison')
plt.ylabel('Count')
plt.xlabel('Images')
plt.xticks(rotation=45)
plt.legend()

# Corrupted bounding boxes comparison
plt.subplot(1, 2, 2)
plt.bar(images, corrupted_file1, alpha=0.6, label='File 1 - Corrupted', width=0.4, align='center')
plt.bar(images, corrupted_file2, alpha=0.6, label='File 2 - Corrupted', width=0.4, align='edge')
plt.title('Corrupted Bounding Boxes Comparison')
plt.ylabel('Count')
plt.xlabel('Images')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()
output_path = "/home/lxt230026/EfficiencyAttack/comparison_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with high resolution

print(f"Plot saved to: {output_path}")
