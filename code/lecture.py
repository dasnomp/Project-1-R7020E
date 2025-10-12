import os
import cv2

# Color images
color_folder = 'datasets/camera_color_image_raw/camera_color_image_raw'
color_images = os.listdir(color_folder)

print(f"Color images: {len(color_images)}")

color_path = os.path.join(color_folder, color_images[0])
color_img = cv2.imread(color_path)

if color_img is not None:
    print(f"Color loaded: {color_img.shape}")
    cv2.imshow('Color', color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: color image")

# Depth images
depth_folder = 'datasets/camera_depth_image_raw/camera_depth_image_raw'
depth_images = os.listdir(depth_folder)

print(f"Depth images: {len(depth_images)}")

depth_path = os.path.join(depth_folder, depth_images[0])
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if depth_img is not None:
    print(f"Depth loaded: {depth_img.shape}")
else:
    print("Error: depth image")