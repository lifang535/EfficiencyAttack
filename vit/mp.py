from multiprocessing import Process, Queue
from torchvision import transforms
from PIL import Image
import sys
import torch
sys.path.append("../captioning")
sys.path.append("../llama3")
sys.path.append("../zeroshot")
from ms_captioning import MSCaptioning
from openai_clip import CLIP
from llama3_api import llama_inference
from tqdm import tqdm
import CONSTANTS
import time

def worker_caption(cropped_list, caption_queue, device):
    ms_caption = MSCaptioning(device=device)
    ms_caption.load_processor_checkpoint()
    ms_caption.load_model()
    
    for cropped_img, idx in tqdm(cropped_list, desc="Processing MS Captioning"):
        caption = ms_caption.inference(cropped_img)
        # If caption contains any tensors, detach them
        if isinstance(caption, torch.Tensor):
            caption = caption.detach().cpu().numpy()
        caption_queue.put((idx, caption))

def worker_clip(cropped_list, clip_queue, device):
    clip = CLIP(device=device)
    clip.load_model()
    clip.load_processor()
    
    for cropped_img, idx in tqdm(cropped_list, desc="Processing CLIP"):
        probs, labels = clip.inference(CONSTANTS.BRID_FAMILY_NAME_LATIN, cropped_img, top_k=3)
        # Detach probs tensor from computation graph and convert to CPU numpy array
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        clip_queue.put((idx, (probs, labels)))
        
def worker_manager(caption_queue, clip_queue, llama_queue, total_items):
    processed_count = 0
    pbar = tqdm(total=total_items, desc="Processing Manager")

    while processed_count < total_items:
        # # 当且仅当两个队列都非空时才处理
        # if not caption_queue.empty() and not clip_queue.empty():
        #     # 获取两个队列的数据
        #     idx, caption = caption_queue.get()
        #     _, clip_data = clip_queue.get()
            
        #     # 直接构建paired_data并放入llama队列
            # paired_data = {
            #     'caption': caption,
            #     'probs': clip_data[0],
            #     'labels': clip_data[1]
            # }
            # llama_queue.put((idx, paired_data))
            # processed_count += 1
            # pbar.update(1)

        # else:
        #     # 如果任一队列为空，短暂休息
        #     time.sleep(0.1)
        

        if not caption_queue.empty():
            idx, caption = caption_queue.get()
            paired_data = {
                'caption': caption,
                'probs': "",
                'labels': ""
            }
            llama_queue.put((idx, paired_data))
            processed_count += 1
            pbar.update(1)
            
        if not clip_queue.empty():
            idx, clip_data = clip_queue.get()
            paired_data = {
                'caption': "",
                'probs': clip_data[0],
                'labels': clip_data[1]
            }
            llama_queue.put((idx, paired_data))
            processed_count += 1
            pbar.update(1)
            
        if caption_queue.empty() and clip_queue.empty():
            time.sleep(0.1)
    pbar.close()

def worker_llama(llama_queue, total_items):
    processed_count = 0
    pbar = tqdm(total=total_items, desc="Processing Llama3")
    
    while processed_count < total_items:  # 使用计数来控制循环
        if not llama_queue.empty():
            idx, data = llama_queue.get()
        
            prompt = CONSTANTS.GET_PROMPT(                
                caption_text=data['caption'],
                probs=data['probs'],
                labels=data['labels'])
            result = llama_inference(prompt)
            
            processed_count += 1
            pbar.update(1)
        else:
            time.sleep(0.1)
    
    pbar.close()
            
            
def concurrent_pipeline(cropped_list, na_cropped_list, device):
    total_items = len(na_cropped_list) + len(cropped_list)
    # 创建队列
    caption_queue = Queue()
    clip_queue = Queue()
    llama_queue = Queue()
    
    # 创建进程
    caption_process = Process(target=worker_caption, 
                            args=(na_cropped_list, caption_queue, device)
                            )
    clip_process = Process(target=worker_clip,
                         args=(cropped_list, clip_queue, device)
                         )
    manager_process = Process(target=worker_manager,
                            args=(caption_queue, clip_queue, llama_queue, total_items)
                            )
    llama_process = Process(target=worker_llama,
                          args=(llama_queue,total_items)
                          )
    
    # 启动所有进程
    caption_process.start()
    clip_process.start()
    manager_process.start()
    llama_process.start()
    
    # 等待所有进程完成
    caption_process.join()
    clip_process.join()
    manager_process.join()
    llama_process.join()  
