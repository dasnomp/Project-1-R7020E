from ultralytics import YOLO
import os
import cv2
import math

# Load the trained model
model = YOLO(r"C:\Users\rano1\Desktop\Project_1\best.pt")

# A4. A3 dimensions in millimeters (real-world size)
A4_WIDTH_MM = 210  # width
A4_HEIGHT_MM = 297  # height
A3_WIDTH_MM = 297  # width
A3_HEIGHT_MM = 420  # height

# Camera parameters 
fx = 1000  
fy = 1000  
cx = 320   
cy = 240   

# Define the function to filter out A4 and A3 printed extinguishers based on size and aspect ratio, 
def filter_extinguishers(result, min_conf=0.05, min_size=(50, 50)):
    keep_boxes, keep_scores, keep_cls = [], [], []
    decoy_boxes = []

    for b, s, c in zip(result.boxes.xyxy.cpu().numpy(),
                       result.boxes.conf.cpu().numpy(),
                       result.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = b.astype(int)
        w, h = x2 - x1, y2 - y1
        
        # Apply the confidence filter
        if s >= min_conf and w >= min_size[0] and h >= min_size[1]:
            # Calculate aspect ratio
            aspect_ratio = h / w

            # Check if aspect ratio is between 1.3 and 1.5 (indicating a decoy object)
            if 1.3 < aspect_ratio < 1.5:
                decoy_boxes.append((x1, y1, x2, y2))  # Mark as decoy
                continue  # Skip if the aspect ratio indicates a decoy

            # Check if bounding box matches A4 or A3 paper size
            if ((0.9 * A4_WIDTH_MM <= w <= 1.1 * A4_WIDTH_MM and 0.9 * A4_HEIGHT_MM <= h <= 1.1 * A4_HEIGHT_MM) or
                (0.9 * A3_WIDTH_MM <= w <= 1.1 * A3_WIDTH_MM and 0.9 * A3_HEIGHT_MM <= h <= 1.1 * A3_HEIGHT_MM)):
                decoy_boxes.append((x1, y1, x2, y2))  # Mark as decoy
                continue  # Skip if it's an A4 or A3 size (likely a printed decoy)

            # If it's not a decoy, we consider it a real extinguisher
            keep_boxes.append([x1, y1, x2, y2])
            keep_scores.append(float(s))
            keep_cls.append(int(c))
    
    return keep_boxes, keep_scores, keep_cls, decoy_boxes

# Path to the directory with test images
color_dir = r"C:\Users\rano1\Desktop\Project_1\raw\test\camera_color_image_raw"
output_dir = r"C:\Users\rano1\Desktop\Project_1\runs\Filtered_out_decoys"

# Create the output directory 
os.makedirs(output_dir, exist_ok=True)

# Run inference on all images in the source folder
for result in model.predict(source=color_dir, iou=0.4, conf=0.15, augment=True, imgsz=768, save=True, stream=True):
    color_path = result.path  # Get the path of the current image
    boxes, scores, clses, decoy_boxes = filter_extinguishers(result)  # Apply filtering function 

    # Read the image to draw bounding boxes
    img = cv2.imread(color_path)
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        # Draw the bounding box and label for detections
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"EXT {score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Save the filtered image with bounding boxes
    save_path = os.path.join(output_dir, os.path.basename(color_path))
    cv2.imwrite(save_path, img)

