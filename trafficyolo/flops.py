from calflops import calculate_flops
from calflops import calculate_flops_hf
import sys
import torch
sys.path.append("..")
from model_zoo import load_from_pretrained
from facenet_pytorch import MTCNN, InceptionResnetV1 # https://github.com/timesler/facenet-pytorch/tree/master
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from fast_plate_ocr import ONNXPlateRecognizer # https://github.com/ankandrew/fast-plate-ocr
import onnxruntime as ort
from torchvision import transforms
from transformers import AutoModelForCausalLM # microsoft/git-base
from transformers import AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(outputchannels=1, aux_loss=True, freeze_backbone=False):
    model = models.segmentation.deeplabv3_resnet101(
        weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
        progress=True, 
        aux_loss=aux_loss)

    if freeze_backbone is True:
        for p in model.parameters():
            p.requires_grad = False
    model.classifier = DeepLabHead(2048, outputchannels)
    return model

def load_seg_model():
    deeplabv3 = create_model()
    checkpoint = torch.load("cmpnt/model_v2.pth", map_location="cpu")
    deeplabv3.load_state_dict(checkpoint['model'])
    return deeplabv3.to(device)

def load_ocr_model():
    onnx_lp_ocr = ONNXPlateRecognizer(
        'global-plates-mobile-vit-v2-model', 
        providers=['CPUExecutionProvider']
        )
    return onnx_lp_ocr

#huggingface
# 136G FLOPs
# OD_MODEL_0, _ = load_from_pretrained(0, device=device)
# OD_MODEL_1, _ = load_from_pretrained(1, device=device)
# OD_MODEL_2, _ = load_from_pretrained(2, device=device)

# 48.5156 GFLOPS
FR_MODEL = InceptionResnetV1(pretrained=f"{"vggface2"}").eval().to(device)

#github model
#to gray scale -> 250,880 per image
# 588.722 GFLOPS
LPR_MODEL = load_seg_model()
OCR_MODEL = load_ocr_model()

#huggingface
#resize -> 1,204,224 FLOPs per image
CAP_MODEL = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(device) 
processor = AutoProcessor.from_pretrained("microsoft/git-base", use_fast=True)
#database
#for face -> 2,562,537 FLOPs
#for plate -> 0 FLOPs

#udp
# 0 FLOPs
def calculate_flops(pipeline, img_num, bbox_num, person_num, car_num):

    # flops of img streaming
    flops_1 = 0.0
    
    # flops of object detection
    flops_2 = img_num * 136e9
    
    # flops of face recognition
    flops_3 = person_num * 6.3e9
    
    # flops of license plate recognition
    flops_4 = car_num * 96.1588e9
    
    # flops of cap
    flops_5 = img_num * 26.7e9
    
    # flops of kr
    flops_6 = person_num * 2562537
    
    if pipeline == 0:
        ret = flops_1 + flops_2 + flops_3 + flops_4 + flops_5 + flops_6
    if pipeline == 1:
        ret = flops_1 + flops_2 + flops_3 + flops_4 + flops_6
    if pipeline == 2:
        ret = flops_1 + flops_2 + flops_5
        
    return ret

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 640, 480).to(device)
    dummy_shape = (1, 3, 640, 480)
    from thop import profile
    from thop import clever_format
    from calflops import calculate_flops
    from torchprofile import profile_macs
    from onnx import load_model
    from onnx_opcounter import calculate_macs
    import onnx
    import onnx_tool
    
    torch.manual_seed(0)

    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    dummy_pixel_values = torch.randn(batch_size, 3, 640, 480)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    flops, macs, params = calculate_flops(model=FR_MODEL, 
                                        # input_shape=(1,197),
                                        input_shape=input_shape,
                                        # input_constructor={"pixel_values": dummy_pixel_values},
                                        output_as_string=True,
                                        output_precision=4,
                                        # transformer_tokenizer=tokenizer,
                                        )
    print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
            
    
    import pynvml
    import time
    def measure_gpu(model, input_token, iter_num, device):
        pynvml.nvmlInit()
        device_id = 0 if device.index is None else device.index
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        t1 = time.time()
        power_list = []
        for _ in range(iter_num):
            model.generate(input_token)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_list.append(power)
        t2 = time.time()
        latency = t2 - t1
        s_energy = sum(power_list) / len(power_list) * latency
        energy = s_energy / (10 ** 6)
        pynvml.nvmlShutdown()
        return latency, energy
    """
            
            import pynvml
            import json
            
            self.count = 0.0
            self.start_time = time.perf_counter()
            pynvml.nvmlInit()
            self.device_id = 0 if self.device.index is None else self.device.index
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        
            self.end_time = time.perf_counter()
            self.time_elapsed = self.end_time - self.start_time
            self.power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            energy = ( self.power * self.time_elapsed ) / (1e6)
            content = {
                "count" : self.count,
                "time" : self.time_elapsed,
                "energy" : self.energy
            }
            with open(self.profile_save_path + ".json", "w") as f:
                json.dump(content, f, indent=4)
    """

    
"""
FR MODEL
------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  27.91 M
fwd MACs:                                                               3.1406 GMACs
fwd FLOPs:                                                              6.3002 GFLOPS
fwd+bwd MACs:                                                           9.4218 GMACs
fwd+bwd FLOPs:                                                          18.9005 GFLOPS

"""

"""
LPR MODEL
------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  60.99 M
fwd MACs:                                                               48.0093 GMACs
fwd FLOPs:                                                              96.1588 GFLOPS
fwd+bwd MACs:                                                           144.028 GMACs
fwd+bwd FLOPs:                                                          288.476 GFLOPS
"""

"""
CAP MODEL
------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  176.62 M
fwd MACs:                                                               13.34 GMACs
fwd FLOPs:                                                              26.7 GFLOPS
fwd+bwd MACs:                                                           40.02 GMACs
fwd+bwd FLOPs:                                                          80.09 GFLOPS
"""