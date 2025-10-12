## Project Structure

```text
Project-1-R7020E/
├── code/
│   ├── train.py              # YOLO training script
│   ├── lecture.py            # Test reading images (color + depth)
│   └── localisation.py       # 3D localization main code
│
├── datasets/                 # NOT on GitHub (download separately)
│   ├── train/                # Training images (Roboflow)
│   ├── valid/                # Validation images (Roboflow)
│   ├── camera_color_image_raw/      # RGB images
│   ├── camera_depth_image_raw/      # Depth images
│   ├── camera_color_camera_info/    # Camera calibration
│   └── camera_depth_camera_info/
│
└── runs/detect/train/weights/
    └── best.pt               
```
