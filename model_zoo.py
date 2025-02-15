import torch
import requests
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

import numpy as np
import random
import pdb
"""
    MODEL ZOO:
    
    Requirements
        1.	The model must be an object detection model.
        2.	The model must be transformer-based.
        3.	It must allow setting $num_queries$ in the model configuration without requiring retraining.
        4.	The model must be publicly accessible and reproducible.

    Due to these considerations, all the models we used are from Hugging Face and belong to the RT-DETR family.
    
    Available Models
        1.  RT_DETR_r50vd [huggingface](https://huggingface.co/PekingU/rtdetr_r50vd) [paper](https://arxiv.org/abs/2304.08069)
        2.  RT_DETR_r50vd_coco_o365 [huggingface](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365) [paper](https://arxiv.org/abs/2304.08069)
        3.  RT_DETR_v2_r50vd [huggingface](https://huggingface.co/PekingU/rtdetr_v2_r50vd) [paper](https://arxiv.org/html/2407.17140v1)
        
    
"""

from transformers import RTDetrForObjectDetection, RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Define a model zoo dictionary
MODEL_ZOO = {
    0: "PekingU/rtdetr_r50vd",
    1: "PekingU/rtdetr_r50vd_coco_o365",
    2: "PekingU/rtdetr_v2_r50vd"
}

def get_model_name(ckpt):
    """Retrieve model name from the zoo given an integer ID or directly validate a string checkpoint."""
    if isinstance(ckpt, int):
        if ckpt not in MODEL_ZOO:
            raise ValueError(f"Invalid checkpoint ID: {ckpt}. Available IDs: {list(MODEL_ZOO.keys())}")
        return MODEL_ZOO[ckpt], ckpt
    elif isinstance(ckpt, str) and ckpt in MODEL_ZOO.values():
        return ckpt, list(MODEL_ZOO.keys())[list(MODEL_ZOO.values()).index(ckpt)]
    else:
        raise ValueError(
            f"Invalid checkpoint: {ckpt}. \n"
            f"Available options:\n"
            f"IDs: {list(MODEL_ZOO.keys())}\n"
            f"Names: {list(MODEL_ZOO.values())}\n"
            "Please choose a valid ID or checkpoint name."
        )

def load_from_pretrained(ckpt=None, device=None):
    """Load a model and image processor from the model zoo."""
    ckpt, ckpt_id = get_model_name(ckpt)

    print(f"Initializing checkpoint: {ckpt}, ID: {ckpt_id}")

    # Load the correct model based on the checkpoint ID
    if ckpt_id in {0, 1}:
        model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device)
    elif ckpt_id == 2:
        model = RTDetrV2ForObjectDetection.from_pretrained(ckpt).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint ID: {ckpt_id}")

    image_processor = RTDetrImageProcessor.from_pretrained(ckpt)
    model = model.eval()
    return model, image_processor
        
if __name__ == "__main__":
    url = "https://farm5.staticflickr.com/4116/4827719363_31f75f0c8f_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thres = 0.25
    num_q = 1000
    for i in range(3):
        model, image_processor = load_from_pretrained(i, device)
        model.config.num_queries = num_q
        
        input = image_processor(images=image, return_tensors="pt").to(device)
        img_tensor = input["pixel_values"]
        target_size = [img_tensor.shape[2:] for _ in range(1)]
        result = model(img_tensor)
        logits = result.logits[0]
        
        output = image_processor.post_process_object_detection(result, 
                                                            threshold = thres, 
                                                            target_sizes = target_size)[0]
        
        print(f"model ID {i}, \nshape: {logits.shape}, num_class{model.config.id2label}")
        print(f"== * ==")
    