import torch
from transformers import AutoProcessor
from PIL import Image
import requests
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from accelerate.test_utils.testing import get_backend
from torchvision import transforms


# checkpoint = "microsoft/git-base"
# processor = AutoProcessor.from_pretrained(checkpoint)


# def transforms(example_batch):
#     images = [x for x in example_batch["image"]]
#     captions = [x for x in example_batch["text"]]
#     inputs = processor(images=images, text=captions, padding="max_length")
#     inputs.update({"labels": inputs["input_ids"]})
#     return inputs


# # train_ds.set_transform(transforms)
# # test_ds.set_transform(transforms)


# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)



url = "https://ik.imagekit.io/tingxi/guitar.JPG?updatedAt=1733902885714"
image = Image.open(requests.get(url, stream=True).raw)


# plt.imshow(image)
# plt.axis('off')  # Turn off the axis
# plt.show()
# plt.savefig("test.png")
# # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
# device, _, _ = get_backend()
# inputs = processor(images=image, return_tensors="pt").to(device)
# pixel_values = inputs.pixel_values.to(device)
# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_caption)

class MSCaptioning:
    def __init__(self, checkpoint="microsoft/git-base", device=None):
        self.device = device
        self.processor = None
        self.model = None
        self.checkpoint = checkpoint
        self.resize_transform = transforms.Compose([transforms.Resize((224, 224)),])
        
    def load_processor_checkpoint(self):
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
    
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint).to(self.device)      
        
    def inference(self, image):
        if isinstance(image, Image.Image):
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values.to(self.device)
        elif isinstance(image, torch.Tensor): 
            pixel_values = image.clone()
        else:
            raise TypeError("input of the ms-captioning model has to be an PIL Image")
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = self.resize_transform(pixel_values)
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption
    
if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_instance = MSCaptioning(device=test_device)
    test_instance.load_processor_checkpoint()
    test_instance.load_model()
    test_output = test_instance.inference(image)
    print(test_output)
    test_instance.transform(image)
    
    