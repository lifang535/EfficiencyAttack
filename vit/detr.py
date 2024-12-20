from transformers import AutoImageProcessor
from transformers import VitDetConfig, VitDetModel
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from transformers import RTDetrConfig, RTDetrModel
import torchvision.transforms as transforms
import torch
from datasets import load_dataset
import dataset
import CONSTANTS
import util
from PIL import Image
import requests
import numpy as np
import json
import pdb
from tqdm import tqdm
import argparse
import overload_attack
import single_attack
import phantom_attack
from datetime import datetime
import os
import random

parser = argparse.ArgumentParser(description="DETR hyperparam setup")
parser.add_argument("--e", type=int, default=-999)
parser.add_argument("--t", type=str, default="infer")
args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("running on : ", device)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model.eval()

to_tensor = transforms.ToTensor()

if __name__ == "__main__":
  results_dict = {}
  # set up MS COCO 2017
  coco_data = load_dataset("detection-datasets/coco", split="val")
  # pdb.set_trace()
  if args.t != "infer":
    # coco_data = coco_data.select(range(CONSTANTS.VAL_SUBSET_SIZE))
    random.seed(42)
    random_indices = random.sample(range(len(coco_data)), CONSTANTS.VAL_SUBSET_SIZE)
    
    coco_data = coco_data.select(random_indices)
    
  if args.t == "infer":
    for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__(), desc="Processing COCO data"):
      # data and ground truth :)
      image_id, image, width, height, bbox_id, category, gt_boxes, area = util.parse_example(example)
      assert len(bbox_id) == len(category) == len(gt_boxes) == len(area)
      
      if image.mode != "RGB":
        image = image.convert("RGB")
          
      inputs = image_processor(images=image, return_tensors="pt").to(device)
      
      with torch.no_grad():
        outputs = model(**inputs)
      target_size = torch.tensor([image.size[::-1]])
      results = image_processor.post_process_object_detection(outputs, 
                                                              threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                              target_sizes = target_size)[0]
      import pdb; pdb.set_trace()
      results = util.move_to_cpu(results)
      pred_scores, pred_labels, pred_boxes = util.parse_prediction(results)
        # for visualization
        # util.visualize_predictions(image, pred_boxes, pred_labels, gt_boxes, category, image_id)

      img_result = util.save_evaluation_to_json(image_id, 
                                                pred_boxes, 
                                                pred_scores,
                                                gt_boxes, 
                                                iou_threshold=0.5)
      
      results_dict[f"image_{image_id}"] = img_result
    
  if args.t == "phantom":
    
    # clean_bbox_num = (pred_scores > 0.9).sum()
    pa = phantom_attack.PhantomAttack(image_list=coco_data,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=args.e,
                                        device=device)
    pa.run()
    results_dict = pa.results_dict

  if args.t == "single":
    sa = single_attack.SingleAttack(image_list=coco_data,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=args.e,
                                        device=device)
    sa.run()
    results_dict = sa.results_dict

  if args.t == "overload":
    
    # clean_bbox_num = (pred_scores > 0.9).sum()
    oa = overload_attack.OverloadAttack(image_list=coco_data,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=args.e,
                                        device=device)
    oa.run()
    results_dict = oa.results_dict
    #TODO: change the overload method accordingly
    # bbox_num = old_overload.attack(model, 
    #                                image_processor, 
    #                                inputs, 
    #                                epochs=args.e,
    #                                device=device)
    
    # results_dict[f"image_{image_id}"] = {"clean_bbox_num": int(clean_bbox_num.item()), "corrupted_bbox_num": bbox_num}
    # pdb.set_trace()
    
    #TODO:
    #saturate gradient -> early stop
    #coefficient
      
  if args.t == "infer":
    output_path = "../prediction/detr_eval_result.json"
    with open(output_path, "w") as f:
      json.dump(results_dict, f, indent=4)

    print(f"Evaluation results saved to {output_path}")
    

  date_str = datetime.now().strftime("%Y%m%d_%H%M")
  output_path = f"../detr-prediction/{date_str}_{args.e}_{args.t}.json"
  with open(output_path, "w") as f:
    json.dump(results_dict, f, indent=4)

  print(f"{args.t} results saved to {output_path}")
    