## Project Structure

```text
Project-1-R7020E/
├── code/
│   ├── train.ipynb           # YOLO training script (Python notebook file)
│   ├── functions.py          # Core pipeline functions for the detection and localization system
│   └── main.py               # Single image pipeline test
│   └── time.py               # Real-time detection simulation
│   └── filteringS.py         # Filtering Sandra
│   └── filteringR.py         # Filtering Rana
│  
├── metrics/
│   ├── images
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
    └── best.pt               # Model weights       
```
