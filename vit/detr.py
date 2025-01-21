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
from datetime import datetime
import os
import random
import time

util.set_all_seeds(0)

parser = argparse.ArgumentParser(description="DETR hyperparam setup")
parser.add_argument("--epoch_num", type=int, default=100)
parser.add_argument("--val_size", type=int, choices=range(1, 4952), default=4952, help="An integer in the range 1-4952 (inclusive)")
parser.add_argument("--algo_name", type=str, default="infer")
parser.add_argument("--pipeline_name", type=str, default=None)
parser.add_argument("--target_cls_idx", type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("running on : ", device)
configuration =  DetrConfig("facebook/detr-resnet-50")
configuration.num_queries = 1000


image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection(configuration).to(device)
# model.config.num_queries = 1000

model.eval()

to_tensor = transforms.ToTensor()

if __name__ == "__main__":
  results_dict = {}
  # set up MS COCO 2017
  coco_data = load_dataset("detection-datasets/coco", split="val")

  random_indices = random.sample(range(len(coco_data)), args.val_size)

  coco_data = coco_data.select(random_indices)
    
  if args.algo_name == "infer":
    for index, example in tqdm(enumerate(coco_data), total=coco_data.__len__(), desc="Processing COCO data"):
      # data and ground truth :)
      image_id, image, width, height, bbox_id, category, gt_boxes, area = util.parse_example(example)
      assert len(bbox_id) == len(category) == len(gt_boxes) == len(area)
      
      if image.mode != "RGB":
        image = image.convert("RGB")
          
      inputs = image_processor(images=image, return_tensors="pt").to(device)
      
      with torch.no_grad():
        start_time = time.perf_counter()
        outputs = model(**inputs)
      target_size = torch.tensor([image.size[::-1]])
      results = image_processor.post_process_object_detection(outputs, 
                                                              threshold = CONSTANTS.POST_PROCESS_THRESH, 
                                                              target_sizes = target_size)[0]
      # import pdb; pdb.set_trace()
      results = util.move_to_cpu(results)
      pred_scores, pred_labels, pred_boxes = util.parse_prediction(results)
        # for visualization
        # util.visualize_predictions(image, pred_boxes, pred_labels, gt_boxes, category, image_id)

      img_result = util.save_evaluation_to_json(image_id, 
                                                pred_boxes, 
                                                pred_scores,
                                                gt_boxes, 
                                                iou_threshold=0.5)
      end_time = time.perf_counter()
      elapsed_time = (end_time - start_time) * 1000
      # results_dict[f"image_{image_id}"] = img_result
      results_dict[f"image_{image_id}"] = {"inference time": round(elapsed_time, 2)}

    
  if args.algo_name == "phantom":
    import phantom_attack

    # clean_bbox_num = (pred_scores > 0.9).sum()
    phantom = phantom_attack.PhantomAttack(model, image_processor,
                                           image_list=coco_data,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=args.epoch_num,
                                        device=device)
    phantom.run()
    results_dict = phantom.results_dict

  if args.algo_name == "single":
    import single_attack

    single = single_attack.SingleAttack(image_list=coco_data,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=args.epoch_num,
                                        device=device)
    single.run()
    results_dict = single.results_dict

  if args.algo_name == "overload":
    import overload_attack

    # clean_bbox_num = (pred_scores > 0.9).sum()
    overload = overload_attack.OverloadAttack(model, image_processor,image_list=coco_data,
                                              image_name_list=None,
                                              img_size=None,
                                              epochs=args.epoch_num,
                                              device=device)
    overload.run()
    results_dict = overload.results_dict
    
  if args.algo_name == "slow":
    import stra_attack
    # raise ValueError("not implemented")
    slow = stra_attack.StraAttack(model, image_processor,
                                  image_list=coco_data,
                                  image_name_list=None,
                                  img_size=None,
                                  epochs=args.epoch_num,
                                  device=device)
    slow.run()
    results_dict = slow.results_dict
    pass
  
  if args.algo_name == "ada":
    import adaptive_attack
    ada = adaptive_attack.SingleAttack(image_list=coco_data,
                                        image_name_list=None,
                                        img_size=None,
                                        epochs=args.epoch_num,
                                        device=device)
    ada.run()
    results_dict = ada.results_dict


  date_str = datetime.now().strftime("%Y%m%d_%H%M")
  output_path = f"../detr-prediction/{args.epoch_num}_{args.algo_name}_{args.pipeline_name}_{args.target_cls_idx}_{args.val_size}.json"
  with open(output_path, "w") as f:
    json.dump(results_dict, f, indent=4)

  print(f"{args.algo_name} results saved to {output_path}")
    