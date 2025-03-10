from multiprocessing import Process, Queue, Value

from configs import config

from module import *

import time

#---------------------------------------------------------------------------
# Create a traffic monitoring pipeline
#---------------------------------------------------------------------------
if __name__ == '__main__':
    frame_queue = Queue()
    target_queue = Queue()
    target_with_description_queue = Queue()
    
    end_signal = Value('b', False)
    
    video_to_frame = VideoToFrame(config, frame_queue)
    object_detection = ObjectDetection(config, frame_queue, target_queue)
    vit_description = VitDescription(config, target_queue, target_with_description_queue)
    frame_to_video = FrameToVideo(config, target_with_description_queue) # llm_summary
    monitor = Monitor(config, frame_queue, target_queue, target_with_description_queue, end_signal)
    
    object_detection.start()
    vit_description.start()
    frame_to_video.start()
    
    time.sleep(60)
    
    video_to_frame.start()
    monitor.start()

    try:
        video_to_frame.join()
        object_detection.join()
        vit_description.join()
        frame_to_video.join()
        
        end_signal.value = True
        monitor.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        video_to_frame.terminate()
        object_detection.terminate()
        vit_description.terminate()
        frame_to_video.terminate()
        monitor.terminate()

    print("[main] Pipeline end!")

