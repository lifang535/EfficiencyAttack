from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

from openai_clip import CLIP

import sys
sys.path.append("../vit")
import CONSTANTS
import torch
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

clip = CLIP(device=torch.device("cuda:0"))
clip.load_model()
clip.load_processor()
probs, labels = clip.inference(CONSTANTS.BRID_FAMILY_NAME_LATIN, image, top_k=3)
print(labels)
print(type(labels))
prob_0 = str(probs[0][0].item())
prob_1 = str(probs[0][1].item())
prob_2 = str(probs[0][2].item())

label_0, label_1, label_2 = labels
print(prob_2)
print(label_2)