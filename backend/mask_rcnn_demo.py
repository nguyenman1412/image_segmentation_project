import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms as T

# COCO category names for reference
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 1. Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Define the image transforms (for input to the model)
transform = T.Compose([
    T.ToTensor(),  # convert to tensor
])

def get_prediction(img_path, threshold=0.5):
    """
    Given an image path, returns:
    - boxes: list of bounding boxes
    - masks: list of binary masks
    - labels: list of class labels
    - scores: list of detection scores
    above the specified threshold.
    """
    # Read the image using OpenCV (BGR)
    img = cv2.imread(img_path)
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transform to tensor
    img_t = transform(img_rgb)
    # Add a batch dimension
    img_t = img_t.unsqueeze(0)
    
    # Model inference
    with torch.no_grad():
        predictions = model(img_t)

    # Extract the predicted data for the first (and only) image in the batch
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()  # shape: [N, 1, H, W]
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    # Filter out objects with score below threshold
    keep_indices = np.where(pred_scores >= threshold)[0]
    
    boxes = pred_boxes[keep_indices].astype(np.int32)
    masks = pred_masks[keep_indices]
    labels = pred_labels[keep_indices]
    scores = pred_scores[keep_indices]

    return boxes, masks, labels, scores, img

def draw_instance_segmentation(img, boxes, masks, labels, scores):
    """
    Draw bounding boxes and segmentation masks on the image.
    Also count how many objects of each class appear.
    """
    # Convert image to BGR for drawing
    img_out = img.copy()
    
    # This dict will store {class_name: count}
    class_counts = {}

    for i in range(len(boxes)):
        box = boxes[i]
        mask = masks[i]
        label_idx = labels[i]
        score = scores[i]
        
        # Convert the mask to a binary mask [H, W]
        # pred_masks are [N, 1, H, W], so we squeeze to remove dim 1
        mask = mask[0]
        mask_binary = mask > 0.5  # threshold the mask

        class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
        
        # Increase the count for this class
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
        
        # Pick a random color for each instance (or you can pick a color by class)
        color = (np.random.randint(0, 255),
                 np.random.randint(0, 255),
                 np.random.randint(0, 255))

        # 1. Draw the bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        # 2. Put the label text above the box
        text = f"{class_name} {score:.2f}"
        cv2.putText(img_out, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. Overlay the mask
        # We'll paint the mask on the image with some alpha
        # Convert mask_binary to uint8 (0 or 255)
        mask_uint8 = (mask_binary * 255).astype(np.uint8)
        # Create a 3-channel version of the mask
        mask_color = np.zeros_like(img_out, dtype=np.uint8)
        mask_color[:, :] = color
        # Combine mask_color and image_out
        # We only add color where mask_uint8 is > 0
        # Let's do a simple blending approach
        img_out[mask_uint8 > 0] = cv2.addWeighted(
            img_out[mask_uint8 > 0], 0.5, 
            mask_color[mask_uint8 > 0], 0.5, 0
        )

    return img_out, class_counts

def main():
    img_path = "backend/street.jpg"  # Change to your image path
    boxes, masks, labels, scores, orig_img = get_prediction(img_path, threshold=0.5)
    # Draw bounding boxes, masks, and get object counts
    result_img, class_counts = draw_instance_segmentation(orig_img, boxes, masks, labels, scores)

    print("Object counts:", class_counts)

    # Show the result
    cv2.imshow("Mask R-CNN Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()