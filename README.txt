# Project-1-R7020E — Fire Extinguisher Detection & 3D Localization

Detect fire extinguishers in RGB images with YOLO, filter out printed decoys using **depth roughness + real size**, and estimate **3D positions** from camera intrinsics and a depth map matched by timestamp.

---

## Team
- *Add names + IDs here*

---

## Repository Structure

```
Project-1-R7020E/
├── code/
│   ├── main.py                     # Entry point (runs detection → filtering → 3D)
│   └── functions.py                # All core functions (YOLO, filtering, depth lookup, 3D)
│
├── datasets/                       # Not tracked in git – place your data here
│   ├── camera_color_image_raw/     # RGB images
│   └── camera_depth_image_raw/     # Depth images
│       └── camera_depth_image_raw/ # (your main.py example points to this subfolder)
│
├── runs/detect/train/weights/
│   └── best.pt                     # YOLO weights (Ultralytics)
│
└── results/                        # Output images saved by the filters (created at run time)
```

> Adjust the folder names to your actual layout if different.

---

## Requirements

- Python 3.9+
- Packages:
  - `ultralytics`
  - `opencv-python`
  - `numpy`

Install (local):
```bash
pip install ultralytics opencv-python numpy
```

**Colab**:
```python
!pip -q install ultralytics opencv-python numpy
```

> Ultralytics will pull PyTorch automatically if needed.

---

## Camera & Depth Assumptions

- **Intrinsics** (in `functions.py`):
  - `fx = 306.0002441`, `fy = 306.112335`
  - `cx = 318.475311`, `cy = 201.369491`
- **Depth units:** raw depth is **millimeters**  
  Set `DEPTH_SCALE = 0.001` (mm → m).  
- **Depth–RGB pairing:** by **nearest timestamp** inside file names (same seconds, then closest nanoseconds).

---

## How It Works (pipeline)

1. **Detect**: `detecter()` runs YOLO on the RGB image and returns detections.
2. **Filter**: choose *one*:
   - `filter_extinguishers(...)` — simple paper-size/ratio filter (no depth).
   - `filter_extinguishers_depth(...)` — **depth-aware**: filters by **real size** (m) and **surface roughness** (median absolute residual after plane fit).
3. **Depth lookup**: `trouver_depth_pour_rgb(...)` finds the closest depth image by timestamp.
4. **3D localization**: `localiser_3d(...)` computes (X, Y, Z) at the bbox center using the pinhole model.

---

## Run

Edit paths in `code/main.py`:

```python
rgb_path = "C:/Users/.../datasets/camera_color_image_raw/camera_color_image_raw/<your_rgb_file>.png"
depth_folder = "camera_depth_image_raw"  # or set to your depth folder root
```

Then run:
```bash
python code/main.py
```

**Switch filtering mode** (in `main.py`):

```python
# Depth-aware filter (recommended):
# detections_valides, image_filtree_path = filter_extinguishers_depth(detections, rgb_img, rgb_path, depth_folder=depth_folder)

# Simple (paper-size) filter:
detections_valides, image_filtree_path = filter_extinguishers(detections, rgb_img, rgb_path)
```

**Outputs**  
- Annotated image saved to `results/<rgb_filename>.png` with:
  - **Green boxes** = kept detections  
  - **Red boxes** = dropped detections (depth mode)  
  - In depth mode, labels show `h=…m  Z=…m  r=…cm` (height, range, roughness)

---

## Key Parameters (in `functions.py`)

```python
# Camera intrinsics (pixels)
fx, fy = 306.0002441, 306.112335
cx, cy = 318.475311, 201.369491

# Depth & thresholds
DEPTH_SCALE = 0.001     # mm → m
MIN_W_M, MIN_H_M = 0.20, 0.70   # min physical size gates (m)
MIN_ASPECT = 1.20       # h/w must be tallish
SHRINK_FRAC = 0.25      # inner crop % per side for depth stats (0.25 = 25%)
ROUGH_THR_M = 0.05      # roughness threshold (meters), ~5 cm
```

**What is `SHRINK_FRAC`?**  
We compute depth statistics on a slightly **smaller inner box** to avoid background.  
- If `SHRINK_FRAC = 0.25`, each side is inset by 25%, so the inner box is roughly **50% of the width and height** of the original.

---

## Notes & Tips

- **Depth loading:** we use `cv2.IMREAD_UNCHANGED` to preserve 16-bit depth values.  
- **Roughness:** computed as **median absolute residual** after fitting a plane to the depth patch (in **meters**).  
  - Flat print: typically few mm  
  - Real extinguisher: can be **2–8 cm** (set `ROUGH_THR_M ≈ 0.05` and tune)
- **Size gating:** width/height in **meters** derived from pixel size, focal length, and median range inside the box.

---

## Troubleshooting

- **No detections**: check your `best.pt` path at `runs/detect/train/weights/best.pt`.  
- **Depth not found**: confirm `depth_folder` points to the correct directory and that file names contain timestamps.
- **Weird sizes**: ensure depth is in **mm** and `DEPTH_SCALE=0.001`. Print a quick sanity check of median depth (should be ~0.5–5.0 m indoors).
- **Ultralytics missing**: `pip install ultralytics` (restart kernel if in Colab).

---

## Example Output (console)

```
[metric] dropped=1 kept=2
Sauvegardé: results/camera_color_image_1727164479.png

RÉSULTATS FINAUX :
  Extincteur 1: X=0.215m, Y=-0.103m, Z=2.741m
  Extincteur 2: X=0.482m, Y=-0.120m, Z=2.763m
```

---

## Acknowledgements

Built for **R7020E – AI and Robotics**. YOLO via **Ultralytics**; image ops via **OpenCV**.
