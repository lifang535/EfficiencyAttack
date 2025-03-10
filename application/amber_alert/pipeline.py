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
    target_with_name_queue = Queue()
    target_with_age_queue = Queue()
    target_with_pose_queue = Queue()
    
    end_signal = Value('b', False)
    
    video_to_frame = VideoToFrame(config, frame_queue)
    object_detection = ObjectDetection(config, frame_queue, target_queue)
    person_recognition = PersonRecognition(config, target_queue, target_with_name_queue)
    age_recognition = AgeRecognition(config, target_with_name_queue, target_with_age_queue)
    pose_recognition = PoseRecognition(config, target_with_age_queue, target_with_pose_queue)

    # violation_detection = ViolationDetection(config, target_with_license_queue, target_with_description_queue)
    frame_to_video = FrameToVideo(config, target_with_pose_queue) # llm_summary
    monitor = Monitor(config, frame_queue, target_queue, target_with_pose_queue, end_signal)
    
    object_detection.start()
    person_recognition.start()
    age_recognition.start()
    pose_recognition.start()
    frame_to_video.start()
    
    time.sleep(10)
    
    video_to_frame.start()
    monitor.start()

    try:
        video_to_frame.join()
        object_detection.join()
        person_recognition.join()
        age_recognition.join()
        pose_recognition.join()
        frame_to_video.join()
        
        end_signal.value = True
        monitor.join()
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt")
        video_to_frame.terminate()
        object_detection.terminate()
        person_recognition.terminate()
        age_recognition.terminate()
        pose_recognition.terminate()
        frame_to_video.terminate()
        monitor.terminate()

    print("[main] Pipeline end!")

