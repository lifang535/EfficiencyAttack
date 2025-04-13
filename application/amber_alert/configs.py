method = 'original_10'

# method = 'before_attacking' # 'before_attacking', 'after_attacking' or 'after_defending'
# method = 'adaptive_attack' 'overload_attack' 'phantom_attack' 'single_attack'

config = {
    'input_video_dir': './input_video',
    
    'output_video_dir': './output_video', # use
    
    'video_number': 1,
    
    'video_start_id': 4,
    
    'input_image_dir': f'../../input/{method}', # use
    
    'output_image_dir': f'../../output/amber_alert/{method}',
    
    'frame_interval': 20000, # 1, # ms
    
    'monitor_interval': 100, # ms
    
    'picture_dir': f'../../picture/amber_alert/{method}',
    'qsize_path': f'../../picture/amber_alert/{method}/qsize.png',
    'latency_path': f'../../picture/amber_alert/{method}/latency.png',
    'times_path': f'../../picture/amber_alert/{method}/times.png',
    'flops_path': f'../../picture/amber_alert/{method}/flops.png',
    
    'frame_size': (224, 224, 3),
    
    'object_detection': {
        'device': 'cuda:3',
        'yolo-tiny_image_processor_path': '../model/yolo-tiny/yolos-tiny_image_processor.pth',
        'yolo-tiny_model_path': '../model/yolo-tiny/yolos-tiny_model.pth',
    },
    
    'yolov5': {
        'device': 'cuda:3',
        'yolov5_path': '../model/yolov5',
        'yolov5n_weights_path': '../model/yolov5/yolov5n.pt',
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'max_det': 10000,
    },
    
    'license_recognition': {
        'device': 'cuda:2',
        'easyocr_model_path': '../model/easyocr/easyocr_model.pth',
    },
    
    'person_recognition': {
        'device': 'cpu',
        'face_recognition_model_path': '../model/face_recognition/face_encodings.pkl',
    },
    
    'age_recognition': {
        'device': 'cuda:1',
        'image_processor_path': '../model/vit-age-classifier/vit-age-classifier_image_processor.pth',
        'model_path': '../model/vit-age-classifier/vit-age-classifier_model.pth',
    },
    
    # gender_recognition': {
    # },
    
    # 'expression_recognition': {
    #     'device': 'cuda:2',
    #     'image_processor_path': '../model/vit-face-expression/vit-face-expression_image_processor.pth',
    #     'model_path': '../model/vit-face-expression/vit-face-expression_model.pth',
    # },
    
    'posture_recognition': {
        'device': 'cuda:0',
        'image_processor_path': '../model/pytorch-openpose/body_pose_image_processor.pth',
        'model_path': '../model/pytorch-openpose/body_pose_model.pth',
    },
}
