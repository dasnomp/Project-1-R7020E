## Project Structure

```text
Project-1-R7020E/
├── code/
│   ├── main.py                     # Entry point (runs detection → filtering → 3D)
│   └── function.py                 # All core functions (YOLO, filtering, depth lookup, 3D, 3D visualization)
│   └── real_time.py                # Real-time detection with video recording
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
