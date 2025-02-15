import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bbox(bbox_data):
    # Create figure and axis
    fig, ax = plt.subplots(1)
    
    # Set the figure size to match typical image dimensions
    fig.set_size_inches(8, 8)
    
    # Colors for different categories
    colors = ['r', 'g', 'b', 'c', 'm']
    
    # Get the axis limits from bbox coordinates
    all_x = [coord[0] for coord in bbox_data['bbox']] + [coord[2] for coord in bbox_data['bbox']]
    all_y = [coord[1] for coord in bbox_data['bbox']] + [coord[3] for coord in bbox_data['bbox']]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Set the axis limits
    ax.set_xlim(x_min - 10, x_max + 10)
    ax.set_ylim(y_max + 10, y_min - 10)  # Reverse y-axis
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    # Draw each bounding box
    for i, bbox in enumerate(bbox_data['bbox']):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=colors[bbox_data['category'][i] % len(colors)],
            facecolor='none'
        )
        
        # Add the patch to the plot
        ax.add_patch(rect)
        
        # Add bbox_id as text
        ax.text(x1, y1-5, f'ID: {bbox_data["bbox_id"][i]}', 
                fontsize=8, color=colors[bbox_data['category'][i] % len(colors)])
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add title
    plt.title('COCO Dataset Bounding Boxes (Image ID: 192)')
    
    # Save the plot
    plt.savefig('coco_bbox_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

# data = {
#     "bbox_id": [478961, 530163, 541526, 545132, 629871],
#     "category": [0, 0, 0, 0, 34],
#     "bbox": [
#         [349.48, 253.48, 481.08, 473.53],
#         [437.21, 218.33, 515.62, 466.36],
#         [0.93, 274.23, 37.46, 479.74],
#         [268.11, 179.46, 379.46, 415.14],
#         [13.2, 381.39, 49.2, 474.37]
#     ],
#     "area": [16061.32005, 10907.634, 4428.93145, 14388.17985, 486.237]
# }
if __name__ == "__main__":
    # Visualize the bounding boxes
    import torch
    torch.set_printoptions(sci_mode=False)  # Disable scientific notation

    import requests
    from PIL import Image
    from transformers import RTDetrImageProcessor, RTDetrForObjectDetection

    url = "http://farm7.staticflickr.com/6016/5960058194_1dfae5d508_z.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = RTDetrImageProcessor.from_pretrained("jadechoghari/RT-DETRv2")
    model = RTDetrForObjectDetection.from_pretrained("jadechoghari/RT-DETRv2").to(device)

    model.config.num_queries = 1000
    model.eval()
    input = image_processor(images=image, return_tensors="pt").to(device)
    img_tensor = input["pixel_values"]
    target_size = [img_tensor.shape[2:] for _ in range(1)]
    with torch.no_grad():
        result = model(img_tensor)
        
    output = image_processor.post_process_object_detection(result, 
                                                           threshold = 0.25, 
                                                           target_sizes = target_size)[0]
    scores, labels, boxes = output["scores"], output["labels"], output["boxes"]
    data = {
        "bbox_id" : [i for i in range(len(scores))],
        "category" : labels.tolist(),
        "bbox" : boxes.tolist(),
    }
    data = {
        "bbox_id": [478961, 530163, 541526, 545132, 629871],
        "category": [0, 0, 0, 0, 34],
        "bbox": [
            [349.48, 253.48, 481.08, 473.53],
            [437.21, 218.33, 515.62, 466.36],
            [0.93, 274.23, 37.46, 479.74],
            [268.11, 179.46, 379.46, 415.14],
            [13.2, 381.39, 49.2, 474.37]
        ],
        "area": [16061.32005, 10907.634, 4428.93145, 14388.17985, 486.237]
    }
    print(len(boxes), boxes.shape)
    print(boxes)
    print(labels)
    visualize_bbox(data)