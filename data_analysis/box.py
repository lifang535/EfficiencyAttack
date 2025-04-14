from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import numpy as np
from tqdm import tqdm
from base_da import get_paths, get_json_paths

base_path = "../results"

def read_bbox(j):
    try:
        with open(j, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Warning] Error parsing JSON file '{j}': {e}")
        return 0

    bbox_area = 0
    count = 0
    for key in data:
        bbox_data = data[key].get("boxes", [])
        for one_bbox in bbox_data:
            if isinstance(one_bbox, list) and len(one_bbox) == 4:
                x1, y1, x2, y2 = one_bbox
                area = (x2 - x1) * (y2 - y1)
                bbox_area += area
                count += 1
    return bbox_area / count if count > 0 else 0


def avg_bbox_area(path):
    json_paths = get_json_paths(path)
    avg_area_list = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(read_bbox, j): j for j in json_paths}
        for future in tqdm(as_completed(futures), total=len(json_paths), desc=f"Processing {path}"):
            avg_area_list.append(future.result())

    return np.mean(avg_area_list) if avg_area_list else 0


if __name__ == "__main__":
    paths = get_paths(base_path)[1]
    all_folder_avg_area = [avg_bbox_area(p) for p in paths]
    print(f"Overall mean bbox area: {np.mean(all_folder_avg_area):.2f}")