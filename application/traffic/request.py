import copy
import numpy as np

class Request(object):
    def __init__(self,
                 video_id: int = None,
                 frame_id: int = None,
                 car_id: int = None,
                 person_id: int = None,
                 frame_number: int = None,
                 car_number: int = None,
                 person_number: int = None,
                 
                 video_fps: int = None,
                 data: np.ndarray = None,   # frame_array
                 box: list = None,          # [x1, y1, x2, y2]
                 label: str = None,
                 signal: int = None,
                 start_time: float = None,
                 times: list = None,
                 flops: list = None,) -> None:
        self.video_id = video_id
        self.frame_id = frame_id
        self.car_id = car_id
        self.person_id = person_id
        self.frame_number = frame_number
        self.car_number = car_number
        self.person_number = person_number
        
        self.video_fps = video_fps
        self.data = data
        self.box = box
        self.label = label
        self.signal = signal
        self.start_time = start_time
        self.times = times
        self.flops = flops

    def copy(self):
        return copy.deepcopy(self)
