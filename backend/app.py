import io
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

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

# Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define image transformation for model input
transform = T.Compose([T.ToTensor()])

app = Flask(__name__)
CORS(app)

def get_prediction_from_array(img, threshold=0.5):
    """
    Given an image (as a NumPy array in BGR format), returns boxes, masks, labels, scores.
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Transform to tensor and add batch dimension
    img_t = transform(img_rgb).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_t)
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()  # shape: [N, 1, H, W]
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    keep_indices = np.where(pred_scores >= threshold)[0]
    boxes = pred_boxes[keep_indices].astype(np.int32)
    masks = pred_masks[keep_indices]
    labels = pred_labels[keep_indices]
    scores = pred_scores[keep_indices]
    return boxes, masks, labels, scores, img

def draw_instance_segmentation(img, boxes, masks, labels, scores):
    """
    Draw bounding boxes and segmentation masks on the image.
    Returns the annotated image and a dictionary of object counts.
    """
    img_out = img.copy()
    class_counts = {}
    
    for i in range(len(boxes)):
        box = boxes[i]
        mask = masks[i][0]  # squeeze out the extra dimension
        mask_binary = mask > 0.5
        
        label_idx = labels[i]
        score = scores[i]
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
        
        # Count objects
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Use a color (you can also map each class to a fixed color)
        color = (np.random.randint(0, 255),
                 np.random.randint(0, 255),
                 np.random.randint(0, 255))
        
        # Draw bounding box and label
        x1, y1, x2, y2 = box
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_out, f"{class_name} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay mask (blend with original image)
        mask_uint8 = (mask_binary * 255).astype(np.uint8)
        mask_color = np.zeros_like(img_out, dtype=np.uint8)
        mask_color[:, :] = color
        img_out[mask_uint8 > 0] = cv2.addWeighted(img_out[mask_uint8 > 0], 0.5,
                                                   mask_color[mask_uint8 > 0], 0.5, 0)
    
    return img_out, class_counts

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        # Read file and decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        boxes, masks, labels, scores, orig_img = get_prediction_from_array(img, threshold=0.5)
        result_img, class_counts = draw_instance_segmentation(orig_img, boxes, masks, labels, scores)
        
        # Encode result image as JPEG and convert to Base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Build a summary string of object counts
        summary = "Detected: " + ", ".join([f"{cls} x {count}" for cls, count in class_counts.items()])
        
        return jsonify({
            "result_image": result_img_b64,
            "object_counts": class_counts,
            "summary": summary
        })
    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)