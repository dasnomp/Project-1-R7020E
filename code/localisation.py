from ultralytics import YOLO
import cv2
import os
import numpy as np

print("="*60)
print("3D LOCALIZATION - FIRE EXTINGUISHER")
print("="*60)

# ===== CAMERA PARAMETERS =====
fx = 306.0002441
fy = 306.112335
cx = 318.475311
cy = 201.369491

print(f"\nCamera parameters:")
print(f"  fx = {fx}")
print(f"  fy = {fy}")
print(f"  cx = {cx}")
print(f"  cy = {cy}")

# ===== LOAD YOLO MODEL =====
print("\nLoading YOLO model...")
model = YOLO('runs/detect/train/weights/best.pt')
print("✅ Model loaded")

# ===== LOAD COLOR IMAGE =====
color_folder = 'datasets/camera_color_image_raw/camera_color_image_raw'
color_images = sorted(os.listdir(color_folder))

color_path = os.path.join(color_folder, color_images[0])
color_img = cv2.imread(color_path)

print(f"\n✅ Color image loaded: {color_img.shape}")

# ===== LOAD DEPTH IMAGE =====
depth_folder = 'datasets/camera_depth_image_raw/camera_depth_image_raw'
depth_images = sorted(os.listdir(depth_folder))

depth_path = os.path.join(depth_folder, depth_images[0])
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

print(f"✅ Depth image loaded: {depth_img.shape}")

# ===== YOLO DETECTION =====
print("\nRunning YOLO detection...")
results = model(color_img)

# ===== PROCESS DETECTIONS =====
detections = results[0].boxes

if len(detections) > 0:
    print(f"\n✅ {len(detections)} fire extinguisher(s) detected!")
    
    for i, box in enumerate(detections):
        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = box.conf[0].cpu().numpy()
        
        # Center of bounding box
        cx_box = int((x1 + x2) / 2)
        cy_box = int((y1 + y2) / 2)
        
        # Get depth at center point (in millimeters)
        Z_mm = depth_img[cy_box, cx_box]
        Z = Z_mm / 1000.0  # Convert to meters
        
        # Calculate 3D position
        X = (cx_box - cx) * Z / fx
        Y = (cy_box - cy) * Z / fy
        
        # Display results
        print(f"\n--- Extinguisher {i+1} ---")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Bounding box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
        print(f"  Center pixel: ({cx_box}, {cy_box})")
        print(f"  3D Position:")
        print(f"    X = {X:.3f} m (left/right)")
        print(f"    Y = {Y:.3f} m (up/down)")
        print(f"    Z = {Z:.3f} m (distance)")
else:
    print("\n❌ No fire extinguisher detected")

print("\n" + "="*60)
print("LOCALIZATION COMPLETE")
print("="*60)