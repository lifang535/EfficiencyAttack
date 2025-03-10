import os
import cv2
import sys
import time
import torch
import pickle
import numpy as np
import face_recognition
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from queue import Empty
from threading import Thread
from matplotlib import pyplot as plt
from torchvision.models import resnet50
from multiprocessing import Process, Queue

from configs import config, method
from request import Request
from llm import handle_text

from pathlib import Path
YOLOV5_FILE = Path(f"../model/yolov5").resolve()
if str(YOLOV5_FILE) not in sys.path:
    sys.path.append(str(YOLOV5_FILE))  # add YOLOV5_FILE to PATH
from models.common import DetectMultiBackend
from utils.general import Profile, non_max_suppression

class VideoToFrame(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue,
                 one_by_one: Queue = None):
        # super(VideoToFrame, self).__init__()
        super().__init__()
        
        self.input_video_dir = config['input_video_dir']
        self.video_number = config['video_number']
        self.video_start_id = config['video_start_id']
        self.frame_interval = config['frame_interval']
        self.frame_size = config['frame_size']
        
        self.input_image_dir = config['input_image_dir']
        
        # to object_detection module
        self.frame_queue = frame_queue
        
        self.end_flag = False
        
        self.one_by_one = one_by_one

    def run(self):
        print(f"[VideoToFrame] Start!")
        
        self._read_image()
        
        self._end()
        
    def _read_image(self):
        input_image_files = os.listdir(self.input_image_dir)
        # sort by image name
        input_image_files.sort(key=lambda x: int(x.split('.')[0])) # input_image_files ?
        
        image_files = sorted([f for f in os.listdir(self.input_image_dir) if f.endswith('.png')])
        
        video_id = 0
        frame_id = 0
        total_frames = len(image_files)
        video_fps = 30
        
        print(f"[VideoToFrame] video_id: {video_id}, total_frames: {total_frames}, video_fps: {video_fps}")
        
        adjust_frame_interval = time.time()

        for image_name in image_files:
            image_path = os.path.join(self.input_image_dir, image_name)

            # 读取图片
            image = cv2.imread(image_path)
            
            # image_array = np.array(image) # lifang535 remove
            image_array = image # lifang535 add
            
            # print(f"image_array: {image_array}") # lifang535 add
            
            
            request = Request(
                video_id=video_id,
                frame_id=frame_id,
                frame_number=total_frames,
                
                video_fps=video_fps,
                data=image_array,
                start_time=time.time(),
                times=[],
                flops=[],
            )
            frame_id += 1
            
            self.frame_queue.put(request)
            
            # time.sleep(1000000) # lifang535 add
            
            time.sleep(max(0, self.frame_interval / 1000 - (time.time() - adjust_frame_interval)))
            adjust_frame_interval = time.time()
            
            if self.one_by_one is not None: # lifang535 add
                _ = self.one_by_one.get()

    def _end(self):
        self.frame_queue.put(None)
        
        self.end_flag = True
        print(f"[VideoToFrame] Stop!")

class ObjectDetection(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue,
                 target_queue: Queue,):
        super().__init__()

        # from video_to_frame module
        self.frame_queue = frame_queue
        # to license_recognition module
        self.target_queue = target_queue
        
        self.frame_size = config['frame_size']
        
        # self.device = torch.device(config['object_detection']['device']) # lifang535 remove
        # self.image_processor_path = config['object_detection']['yolo-tiny_image_processor_path']
        # self.model_path = config['object_detection']['yolo-tiny_model_path']
        
        self.device = torch.device("cuda:1") # lifang535 add
        self.yolov5n_weights_path = config['yolov5']['yolov5n_weights_path']
        self.conf_thres = config['yolov5']['conf_thres']
        self.iou_thres = config['yolov5']['iou_thres']
        self.max_det = config['yolov5']['max_det']
        self.target_size = None
        
        # self.monitor_interval = config['monitor_interval'] # lifang535
        
        self.image_processor = None
        self.model = None
        self.id2label = None
        
        # self.thread_pool = ThreadPoolExecutor(max_workers=1000)
        
        self.end_flag = False

    def run(self):
        print(f"[ObjectDetection] Start!")
        
        # self.image_processor = torch.load(self.image_processor_path, map_location=self.device) # lifang535 remove
        # self.model = torch.load(self.model_path, map_location=self.device)
        # self.id2label = self.model.config.id2label
        
        self.model = DetectMultiBackend(weights=self.yolov5n_weights_path, device=self.device) # lifang535 add
        self.id2label = self.model.names
        # image_size = (480, 640)
        # self.model.warmup(imgsz=(1, 3, *image_size))  # warmup
        
        while not self.end_flag:
            try:
                request = self.frame_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            # temp_thread = self.thread_pool.submit(self._infer, request)
            
            self._infer(request)
            
            # if request.frame_id % 50 == 0: # lifang535 remove
            if request.frame_id % 1 == 0: # lifang535 add: new app
                print(f"[ObjectDetection] video_id: {request.video_id}, frame_id: {request.frame_id}")
    
    def _infer(self, request):
        def _preprocess(image_array): # lifang535 add
            image_array = image_array.transpose((2, 0, 1))[::-1]
            image_array = np.ascontiguousarray(image_array)
            image_tensor = torch.from_numpy(image_array).to(self.device).float()
            image_tensor /= 255.0
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor[None]
            
            # print(f"[ObjectDetection] image_tensor = {image_tensor}")
            
            return image_tensor
        
        def _postprocess(outputs): # with batch size 1 # lifang535 add
            outputs = outputs[0].unsqueeze(0)
            outputs = non_max_suppression(prediction=outputs, conf_thres=self.conf_thres, iou_thres=self.iou_thres, max_det=self.max_det)
            results = []
            for output in outputs:
                result = {
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                }
                # print(f"[ObjectDetection] len(output) = {len(output)}")
                # time.sleep(1000000)
                if len(output):
                    for *xyxy, conf, cls in reversed(output):
                        c = int(cls)
                        label = f"{self.id2label[c]}"
                        confidence = float(conf)
                        box = [float(i) for i in xyxy] # TODO: 0 ~ 1
                        result["boxes"].append(box)
                        result["scores"].append(confidence)
                        result["labels"].append(label)
                results.append(result)
            return results
        
        flops, params = 0, 0
        
        frame_array = request.data
        
        # frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB) # lifang535 remove
        # # inputs = self.image_processor(images=[frame_array], return_tensors="pt").to(self.device)
        # inputs = {
        #     'pixel_values': (torch.from_numpy(frame_array.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0).to(self.device)
        # }
        
        inputs = _preprocess(frame_array) # lifang535 add
        
        with torch.no_grad():
            # outputs = self.model(**inputs)
            # outputs = self.model(inputs['pixel_values']) # lifang535 remove
            outputs = self.model(inputs)

        results = _postprocess(outputs) # lifang535 add
        
        for i, result in enumerate(results): # lifang535 remove
            # car_number = sum([1 for label in result["labels"] if self.id2label[label.item()] == 'car']) # lifang535 remove
            # person_number = sum([1 for label in result["labels"] if self.id2label[label.item()] == 'person']) # lifang535 remove
            # car_number = sum([1 for label in result["labels"] if label == 'car']) # lifang535 add
            target_label_list = ['car']
            car_number = sum([1 for label in result["labels"] if label in target_label_list])
            
            print(f"[ObjectDetection] car_number = {car_number}")
            
            request.car_number = car_number
            request.times.append(time.time())
            request.flops.append(flops)
            
            self.target_queue.put(request)
            
            car_id = 0
            person_id = 0
            
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                # box = [round(i, 5) for i in box.tolist()] # lifang535 remove
                box = [round(i, 5) for i in box]
                if label in target_label_list: # lifang535 add: 电器
                    req = request.copy()
                    req.car_id = car_id
                    
                    req.box = box
                    # req.label = self.id2label[label.item()] # lifang535 remove
                    req.label = label # lifang535 add
                    self.target_queue.put(req)
                    
                    car_id += 1
                    
                # print(
                #     f"Detected {self.id2label[label.item()]} with confidence "
                #     f"{round(score.item(), 3)} at location {box}"
                # )

    def _end(self):
        self.target_queue.put(None)
        
        self.end_flag = True
        print(f"[ObjectDetection] Stop!")

class LicenseRecognition(Process):
    def __init__(self, 
                 config: dict,
                 target_queue: Queue,
                 target_with_license_queue: Queue):
        super().__init__()
        
        # from object_detection module
        self.target_queue = target_queue
        # to frame_to_video module
        self.target_with_license_queue = target_with_license_queue
        
        # self.device = torch.device(config['license_recognition']['device'])
        self.device = torch.device("cuda:2")
        self.model_path = config['license_recognition']['easyocr_model_path']
        
        self.monitor_interval = config['monitor_interval']
        
        self.model = None
        
        self.end_flag = False

    def run(self):
        print(f"[LicenseRecognition] Start!")
        
        self.model = torch.load(self.model_path)
        self.model.device = self.device
        
        while not self.end_flag:
            try:
                request = self.target_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            self._infer(request)
            
            # print(f"[LicenseRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}")
            
    def _infer(self, request):
        flops, params = 0, 0
        
        if request.box is not None:
            frame_array = request.data
            
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

            frame_size = frame_array.shape
            
            box = request.box
            
            # Relative coordinates need to be converted to absolute coordinates
            x1, y1, x2, y2 = box
            # x1 = int(x1 * frame_size[1]) # lifang535 remove
            # y1 = int(y1 * frame_size[0])
            # x2 = int(x2 * frame_size[1])
            # y2 = int(y2 * frame_size[0])
            x1 = int(x1) # lifang535 add
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            inputs = frame_array[y1:y2, x1:x2]
            
            # try: # To measure the FLOPs and parameters of the model        
            #     inputs_array = np.array(inputs.copy().transpose(2, 0, 1), dtype=np.float32)
            #     inputs_tensor = torch.from_numpy(inputs_array).unsqueeze(0).to("cuda:0")
            #     flops, params = profile(self.model.detector.module, inputs=(inputs_tensor, )) # add，好像有点问题，在下面的 try 语句会报错
            # except Exception as e:
            #     pass

            with torch.no_grad():
                try: # add
                    outputs = self.model.readtext(inputs)
                except Exception as e:
                    print(f"[LicenseRecognition] Error: {e}")
                    print(f"[LicenseRecognition] inputs: {inputs.shape}")
                    outputs = []
                
            results = outputs
            
            label = "none"
            
            # select the highest confidence result
            if len(results) > 0:
                sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
                label = sorted_results[0][1]

            request.label += f": {label}"
            
            # print(f"[LicenseRecognition] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}, label: {request.label}")
        
        request.times.append(time.time())
        request.flops.append(flops)
        
        self.target_with_license_queue.put(request)
        pass
            
    def _end(self):
        self.target_with_license_queue.put(None)
        
        self.end_flag = True
        print(f"[LicenseRecognition] Stop!")

        
class ViolationDetection(Process):
    def __init__(self, 
                 config: dict,
                 target_with_license_queue: Queue,
                 target_with_description_queue: Queue):
        super().__init__()
        
        # from object_detection module
        self.target_with_license_queue = target_with_license_queue
        # to frame_to_video module
        self.target_with_description_queue = target_with_description_queue
        
        self.device = torch.device("cuda:3")
        
        self.monitor_interval = config['monitor_interval']
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self.end_flag = False

    def run(self):
        print(f"[ViolationDetection] Start!")
        from torchvision import datasets, models
        import torchvision.transforms as transforms
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        print(f"[ViolationDetection] loaded model 1")
        
        self.model.to(self.device)
        # self.model.device = self.device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),             # 缩放图像
            transforms.CenterCrop(224),          # 中心裁剪为224x224
            transforms.ToTensor(),               # 转换为Tensor
            transforms.Normalize(               # 正则化：用ImageNet的均值和标准差
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"[ViolationDetection] loaded model 2")
        
        while not self.end_flag:
            try:
                request = self.target_with_license_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self._end()
                break
            
            self._infer(request)
            
            # print(f"[ViolationDetection] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}")
            
    def _infer(self, request):
        flops, params = 0, 0
        
        if request.box is not None:
            frame_array = request.data
            
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

            frame_size = frame_array.shape
            
            box = request.box
            
            # Relative coordinates need to be converted to absolute coordinates
            x1, y1, x2, y2 = box
            # x1 = int(x1 * frame_size[1]) # lifang535 remove
            # y1 = int(y1 * frame_size[0])
            # x2 = int(x2 * frame_size[1])
            # y2 = int(y2 * frame_size[0])
            x1 = int(x1) # lifang535 add
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            # inputs = frame_array[y1:y2, x1:x2]
            inputs = frame_array

            with torch.no_grad():
                try: # add
                    inputs = self.transform(inputs).unsqueeze(0).to(self.device)
                    # print(f"[ViolationDetection] inputs: {inputs}")
                    # 使用模型生成图像描述
                    with torch.no_grad():
                        output = self.model(inputs)

                    # 解码生成的文本
                    _, predicted_class = torch.max(output, 1)
                    label = f"{predicted_class.item()}"
                    
                except Exception as e:
                    print(f"[ViolationDetection] Error: {e}")
                    print(f"[ViolationDetection] inputs: {inputs}")
                    label = ""
                
            results = label
            
            label = "none" if len(results) == 0 else results
            
            # # select the highest confidence result
            # if len(results) > 0:
            #     sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
            #     label = sorted_results[0][1]
            #     print(f"[ViolationDetection] label: {label}") # TODO: TEST 应该打印不出来，因为没有电器

            request.label += f": {label}"
            print(f"[ViolationDetection] label: {request.label}")
            
            # print(f"[ViolationDetection] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}, label: {request.label}")
        
        request.times.append(time.time())
        request.flops.append(flops)
        
        self.target_with_description_queue.put(request)
        pass
            
    def _end(self):
        self.target_with_description_queue.put(None)
        
        self.end_flag = True
        print(f"[ViolationDetection] Stop!")

class FrameToVideo(Process):
    def __init__(self, 
                 config: dict,
                 target_with_description_queue: Queue,
                 one_by_one: Queue = None):
        super().__init__()
        
        self.output_video_dir = config['output_video_dir']
        self.output_image_dir = config['output_image_dir']
        if not os.path.exists(self.output_video_dir):
            os.makedirs(self.output_video_dir)
        if not os.path.exists(self.output_image_dir):
            os.makedirs(self.output_image_dir)
        
        # from license_recognition module
        self.target_with_description_queue = target_with_description_queue
        
        # only used to send end signal
        self.frame_queue = Queue()
        self.video_queue = Queue()
        
        # save the frames of each video
        self.video_dict = {}
        self.video_saved_set = set()
        
        self.llm = None
        
        # save the process time of each frame
        self.process_time_dict = {} # add
        self.times = {} # add
        self.times = { # add
            'od': [],
            'lr': [],
            'pr': [],
            'resnet': [],
        }
        
        self.flops = {} # add
        self.car_number = {}
        self.person_number = {}
        
        self.picture_dir = config['picture_dir']
        if not os.path.exists(self.picture_dir):
            os.makedirs(self.picture_dir)
        self.latency_path = config['latency_path']
        self.times_path = config['times_path']
        self.flops_path = config['flops_path']

        self.end_flag = False
        
        self.one_by_one = one_by_one

    def run(self):
        print(f"[FrameToVideo] Start!")

        car_get_thread = Thread(target=self.car_get)

        car_get_thread.start()
        try:
            car_get_thread.join()
        except KeyboardInterrupt:
            pass
        
        print(f"[FrameToVideo] Stop!")
            
    def car_get(self):
        while not self.end_flag:
            try:
                request = self.target_with_description_queue.get(timeout=1)
            except Empty:
                continue
            
            if request is None:
                self.frame_queue.put(None)
                break
            
            # print(f"[FrameToVideo] video_id: {request.video_id}, frame_id: {request.frame_id}, car_id: {request.car_id}")
            
            video_id = request.video_id
            frame_id = request.frame_id
            car_id = request.car_id
            
            if video_id not in self.video_dict:
                self.video_dict[video_id] = {}
            if frame_id not in self.video_dict[video_id]:
                self.video_dict[video_id][frame_id] = {}
                self.video_dict[video_id][frame_id]['request'] = request
                
                self.video_dict[video_id][frame_id]['car'] = {}
                self.video_dict[video_id][frame_id]['person'] = {}
                self.video_dict[video_id][frame_id]['times'] = {'od': request.times[0] - request.start_time, 'lr': request.times[1] - request.start_time, 'resnet': request.times[2] - request.start_time}
                self.video_dict[video_id][frame_id]['flops'] = {'od': request.flops[0], 'lr': 0, 'resnet': 0}
            
            self.video_dict[video_id][frame_id]['times']['lr'] = request.times[1] - request.start_time
            self.video_dict[video_id][frame_id]['flops']['lr'] += request.flops[1]
            
            self.video_dict[video_id][frame_id]['times']['resnet'] = request.times[2] - request.start_time
            self.video_dict[video_id][frame_id]['flops']['resnet'] += request.flops[1]
            
            if request.box is not None:
                self.video_dict[video_id][frame_id]['car'][car_id] = {'box': request.box, 'label': request.label}
            
            # if len(self.video_dict[video_id]) == request.frame_number: # 删掉这行是为了记录 self.process_time_dict[frame_id]（认为顺序发送不可能乱序到达下游模块）
            # self.check_video(video_id)
            
            self.check_frame(video_id, frame_id)
            
    def _infer(self, labels):
        # LLM
        result = handle_text(f"{labels}")
        print(f"[FrameToVideo] result: {result}")
        return result
        
            
    def check_frame(self, video_id, frame_id):
        request = self.video_dict[video_id][frame_id]['request']
        frame_number = request.frame_number
        
        car_number = request.car_number
        person_number = request.person_number
        
        print(f"[FrameToVideo] video_id: {video_id}, frame_id: {frame_id}, car_number: {car_number}, len(['car']): {len(self.video_dict[video_id][frame_id]['car'])}")
        if len(self.video_dict[video_id][frame_id]['car']) == car_number:
            # 000001.png, 000002.png, ..., 000010.png, ..., 000100.png, ..., 001000.png, ...
            frame = request.data # numpy.ndarray
            
            labels = []
            
            for car_id, car in self.video_dict[video_id][frame_id]['car'].items():
                box = car['box']
                label = car['label']
                
                # Relative coordinates need to be converted to absolute coordinates
                x1, y1, x2, y2 = box
                # x1 = int(x1 * frame_size[1]) # lifang535 remove
                # y1 = int(y1 * frame_size[0])
                # x2 = int(x2 * frame_size[1])
                # y2 = int(y2 * frame_size[0])
                x1 = int(x1) # lifang535 add
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                # print(f"[FrameToVideo] Car box: ({x1}, {y1}, {x2}, {y2}) label {label}")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                labels.append(label)
            
            frame_path = os.path.join(self.output_image_dir, f"{(frame_id + 1):06d}.png")
            cv2.imwrite(frame_path, frame)

            self.process_time_dict[frame_id] = time.time() - request.start_time # add，TODO: 考虑顺序
            self.car_number[frame_id] = car_number
            self.person_number[frame_id] = person_number

            self.times['od'].append(self.video_dict[video_id][frame_id]['times']['od'])
            self.times['lr'].append(self.video_dict[video_id][frame_id]['times']['lr'])
            self.times['resnet'].append(self.video_dict[video_id][frame_id]['times']['resnet'])

            # self.video_dict[video_id].pop(frame_id)
            print(f"[FrameToVideo] video_id: {video_id}, frame_id: {frame_id}, saved in {frame_path}")

            if frame_id == frame_number - 1:
                self._end()

    def draw_latency(self): # add，会调用两次
        process_time_list = [self.process_time_dict[frame_id] for frame_id in range(len(self.process_time_dict))]
        car_number_list = [self.car_number[frame_id] for frame_id in range(len(self.car_number))]
        person_number_list = [self.person_number[frame_id] for frame_id in range(len(self.person_number))]
        
        print(f"[FrameToVideo] process_time_list = {process_time_list}")
        print(f"[FrameToVideo] car_number_list = {car_number_list}")
        print(f"[FrameToVideo] person_number_list = {person_number_list}")
        
        # 记录在 logs.txt 中（续写）
        with open('logs.txt', 'a') as f:
            f.write(f"Method = {method}\n")
            f.write(f"process_time_list = {process_time_list}\n")
            f.write(f"car_number_list = {car_number_list}\n")
            f.write(f"person_number_list = {person_number_list}\n")
            f.write('\n')
        
        process_time_list = process_time_list[2:]
        car_number_list = car_number_list[2:]
        person_number_list = person_number_list[2:]
        
        # 创建一个包含 3 个子图的图形
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # 绘制第一个折线图
        axs[0].plot(process_time_list, label='Process Time')
        axs[0].set_title('Process Time Over Frames')
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel('Process Time')
        axs[0].legend()

        # 绘制第二个折线图
        axs[1].plot(car_number_list, label='Car Number', color='orange')
        axs[1].set_title('Car Number Over Frames')
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('Car Number')
        axs[1].legend()

        # 绘制第三个折线图
        axs[2].plot(person_number_list, label='Person Number', color='green')
        axs[2].set_title('Person Number Over Frames')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('Person Number')
        axs[2].legend()

        # 调整子图之间的间距
        plt.tight_layout()

        # 保存图形为 PDF 文件
        # latency_path = '../picture/latency/before_attacking.png'
        plt.savefig(self.latency_path)
        print(f"[FrameToVideo] saved latency plot to {self.latency_path}")
        
        # 关闭图形
        plt.close()
    
        # plt.plot(range(len(process_time_list)), process_time_list)
        # plt.savefig('../latency/2.pdf')
    
    def draw_times(self):
        print(f"[FrameToVideo] self.times = {self.times}")
        
        # 记录在 logs.txt 中（续写）
        with open('logs.txt', 'a') as f:
            f.write(f"Method = {method}\n")
            f.write(f"times = {self.times}\n")
            f.write('\n\n')
        
        # 画在一张图上
        # 创建一个图形
        plt.figure(figsize=(10, 5))
        
        # 绘制折线图
        plt.plot(self.times['od'][2:], label='Object Detection')
        plt.plot(self.times['lr'][2:], label='License Recognition')
        plt.plot(self.times['pr'][2:], label='Person Recognition')
        
        # 添加标题和标签
        plt.title('Inference Time Over Frames')
        plt.xlabel('Frame')
        plt.ylabel('Inference Time')
        
        # 添加图例
        plt.legend()

        # 保存图形为 PDF 文件
        plt.savefig(self.times_path)
        print(f"[FrameToVideo] saved times plot to {self.times_path}")
        
        # 关闭图形
        plt.close()
        
    def draw_flops(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        axs[0].plot(self.flops['od'], label='Object Detection')
        axs[0].set_title('FLOPs Over Frames')
        axs[0].set_xlabel('Frame')
        axs[0].set_ylabel('FLOPs')
        axs[0].legend()
        
        axs[1].plot(self.flops['lr'], label='License Recognition', color='orange')
        axs[1].set_title('FLOPs Over Frames')
        axs[1].set_xlabel('Frame')
        axs[1].set_ylabel('FLOPs')
        axs[1].legend()
        
        axs[2].plot(self.flops['pr'], label='Person Recognition', color='green')
        axs[2].set_title('FLOPs Over Frames')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylabel('FLOPs')
        axs[2].legend()
        
        plt.tight_layout()
        
        plt.savefig(self.flops_path)
        print(f"[FrameToVideo] saved flops plot to {self.flops_path}")
        
        plt.close()

    def _end(self):
        self.draw_latency() # add
        self.draw_times() # add
        
        # self.draw_flops() # add
        
        self.end_flag = True

class Monitor(Process):
    def __init__(self, 
                 config: dict,
                 frame_queue: Queue,
                 car_queue: Queue,
                 person_queue: Queue,
                 end_flag):
        super().__init__()

        # ObjectDetection module
        self.frame_queue = frame_queue
        # LicenseRecognition module
        self.car_queue = car_queue
        # PersonRecognition module
        self.person_queue = person_queue
        
        self.monitor_interval = config['monitor_interval']
        
        self.qsize_path = config['qsize_path']
        
        self.end_flag = end_flag

    def run(self):
        print(f"[Monitor] Start!")
        
        frame_qsize = []
        car_qsize = []
        person_qsize = []
        
        adjust_monitor_interval = time.time()
        while not self.end_flag.value:
            frame_qsize.append(self.frame_queue.qsize())
            car_qsize.append(self.car_queue.qsize())
            person_qsize.append(self.person_queue.qsize())
            
            time.sleep(max(0, self.monitor_interval / 1000 - (time.time() - adjust_monitor_interval)))
            adjust_monitor_interval = time.time()
            
        # draw queue size
        # 创建一个包含 3 个子图的图形
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # 绘制第一个折线图
        axs[0].plot(frame_qsize, label='frame qsize')
        axs[0].set_title('ObjectDetection')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Qsize')
        axs[0].legend()

        # 绘制第二个折线图
        axs[1].plot(car_qsize, label='car qsize', color='orange')
        axs[1].set_title('LicenseRecognition')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Qsize')
        axs[1].legend()

        # 绘制第三个折线图
        axs[2].plot(person_qsize, label='person qsize', color='green')
        axs[2].set_title('PersonRecognition')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Qsize')
        axs[2].legend()

        # 调整子图之间的间距
        plt.tight_layout()

        # 保存图形为 PDF 文件
        plt.savefig(self.qsize_path)
        print(f"[Monitor] saved qsize plot to {self.qsize_path}")
        
        # 关闭图形
        plt.close()
        
    def _end(self):
        self.end_flag = True
        print(f"[Monitor] Stop!")
