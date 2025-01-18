from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

class CLIP:
  def __init__(self, device):
    self.device = device
    self.model = None
    self.processor = None
    
  def load_model(self, model_path="openai/clip-vit-large-patch14"):
    if self.model == None:
      self.model = CLIPModel.from_pretrained(model_path).to(self.device)
    else:
      Warning("CLIP model is not None, reloading now")
    return True
  
  def load_processor(self, processor_path="openai/clip-vit-large-patch14"):
    if self.processor == None:
      self.processor = CLIPProcessor.from_pretrained(processor_path)
    else:
      Warning("processor is not None, reloading now")
    return True
  
  def inference(self, text, image, top_k=3, return_type="pt", if_padding=True):
    inputs = self.processor(text, image, return_tensors=return_type, padding=if_padding)
    outputs = self.model(**inputs.to(self.device))
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    top_probs, top_indices = probs.topk(top_k, dim=1)
    top_labels = [text[idx] for idx in top_indices[0].tolist()]
    return top_probs, top_labels

  