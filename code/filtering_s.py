#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =======================
# CONFIG — set paths here
# =======================
COLOR_DIR   = "test_files/test/test/camera_color_image_raw"
DEPTH_DIR   = "test_files/test/test/camera_depth_image_raw"
WEIGHTS     = "best.pt"

# Camera intrinsics (pixels)
FX = 306.00024
FY = 306.11234

# Depth scale: raw → meters (0.001 if mm; 1.0 if meters)
DEPTH_SCALE = 0.001

# YOLO
YOLO_IMGSZ = 768
YOLO_CONF  = 0.15
YOLO_IOU   = 0.40
YOLO_AUG   = False

# Filtering thresholds (meters)
MIN_W_M     = 0.20
MIN_H_M     = 0.70
MIN_ASPECT  = 1.20
SHRINK_FRAC = 0.08
ROUGH_THR_M = 0.10

# Output
FPS           = 4
VIDEO_FOURCC  = "mp4v"               # try 'avc1' if needed
VIDEO_OUT     = "metrics/stream.mp4"
FRAMES_DIR    = "metrics/frames"

# Mapping json will be kept next to this script:
# code/color_depth_mapping.json  (auto-created if missing)
# =======================

# --- standard libs only up here ---
import re, json, time, sys, subprocess, importlib
from bisect import bisect_left
from pathlib import Path

# ---------- auto-install deps if missing ----------
def ensure_pkg(mod_name: str, pip_name: str = None):
    """
    Try to import a module; if it fails, pip install it, then import again.
    """
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        pkg = pip_name or mod_name
        print(f"[setup] '{mod_name}' not found; installing '{pkg}' ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install {pkg}. Check internet/permissions.") from e
        # try import again
        return importlib.import_module(mod_name)

# Ensure core deps
np = ensure_pkg("numpy")
cv2 = ensure_pkg("cv2", "opencv-python")
ultra_mod = ensure_pkg("ultralytics")
from ultralytics import YOLO  # type: ignore

# ---------- script paths ----------
from pathlib import Path

try:
    # running as a .py script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # running in a notebook/REPL
    SCRIPT_DIR = Path.cwd()

#SCRIPT_DIR = Path(__file__).resolve().parent
MAPPING_JSON = SCRIPT_DIR / "color_depth_mapping.json"

# ---------- IO utils ----------
def natural_key(s: str):
    parts = re.split(r'(\d+)', s)
    return [int(t) if t.isdigit() else t.lower() for t in parts]

def sorted_files(p: Path):
    files = [x for x in Path(p).iterdir() if x.is_file()]
    files.sort(key=lambda f: natural_key(f.name))
    return files

def load_color(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    return im  # BGR

def load_depth(p: Path):
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(p)
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d  # single-channel

def draw_boxes(img_bgr, boxes_xyxy, color, thickness=2, labels=None):
    out = img_bgr.copy()
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(out, labels[i], (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out

# ---------- Geometry / filtering ----------
def shrink_box(x1, y1, x2, y2, W, H, margin=0.08):
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    dx, dy = int(bw * margin), int(bh * margin)
    nx1 = max(0, x1 + dx); ny1 = max(0, y1 + dy)
    nx2 = min(W - 1, x2 - dx); ny2 = min(H - 1, y2 - dy)
    if nx2 <= nx1 or ny2 <= ny1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2

def plane_residual_mad(depth_patch_m):
    dp = depth_patch_m.astype(np.float32)
    valid = (dp > 0) & np.isfinite(dp)
    if valid.sum() < 200:
        return np.nan
    H, W = dp.shape
    vv, uu = np.mgrid[0:H, 0:W]
    Z = dp[valid].reshape(-1, 1)
    U = uu[valid].reshape(-1, 1)
    V = vv[valid].reshape(-1, 1)
    X = np.hstack([U, V, np.ones_like(U)])
    coef, *_ = np.linalg.lstsq(X, Z, rcond=None)
    Zhat = (X @ coef).ravel()
    res = np.abs(Z.ravel() - Zhat)
    mad = np.median(np.abs(res - np.median(res)))
    return float(mad)

def filter_boxes_size_roughness_metric(
    boxes_xyxy, depth_img_raw, fx, fy, depth_scale,
    min_w_m=0.2, min_h_m=0.7, min_aspect=1.2, shrink=0.08, roughness_thr=0.1
):
    H, W = depth_img_raw.shape[:2]
    kept, dropped = [], []
    for (x1, y1, x2, y2) in boxes_xyxy.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        ix1, iy1, ix2, iy2 = shrink_box(x1, y1, x2, y2, W, H, margin=shrink)
        dep_patch_raw = depth_img_raw[iy1:iy2 + 1, ix1:ix2 + 1].astype(np.float32)
        valid = dep_patch_raw[(dep_patch_raw > 0) & np.isfinite(dep_patch_raw)]
        if valid.size < 50:
            dropped.append(((x1, y1, x2, y2), "no_depth")); continue

        Z = float(np.median(valid)) * depth_scale  # meters
        if not np.isfinite(Z) or Z <= 0:
            dropped.append(((x1, y1, x2, y2), "invalid_Z")); continue

        w_px = x2 - x1 + 1; h_px = y2 - y1 + 1
        width_m  = (w_px * Z) / float(fx)
        height_m = (h_px * Z) / float(fy)
        aspect_m = height_m / max(width_m, 1e-9)

        if width_m < min_w_m:
            dropped.append(((x1, y1, x2, y2), f"width<{min_w_m:.2f}m ({width_m:.2f})")); continue
        if height_m < min_h_m:
            dropped.append(((x1, y1, x2, y2), f"height<{min_h_m:.2f}m ({height_m:.2f})")); continue
        if aspect_m < min_aspect:
            dropped.append(((x1, y1, x2, y2), f"aspect<{min_aspect:.2f} ({aspect_m:.2f})")); continue

        rough = plane_residual_mad(dep_patch_raw * depth_scale)
        if np.isnan(rough):
            dropped.append(((x1, y1, x2, y2), "rough_nan")); continue
        if rough < roughness_thr:
            dropped.append(((x1, y1, x2, y2), f"rough<{roughness_thr:.3f}m ({rough:.3f})")); continue

        kept.append({
            "box": (x1, y1, x2, y2),
            "width_m": width_m, "height_m": height_m, "Z_m": Z,
            "aspect_m": aspect_m, "rough_mad_m": rough, "reason": "ok"
        })
    return kept, dropped

# ---------- Mapping by timestamp in filename ----------
def parse_last_int_in_name(name: str):
    stem = Path(name).stem
    m = re.findall(r'(\d+)', stem)
    return int(m[-1]) if m else None

def build_timestamp_mapping(color_dir: Path, depth_dir: Path, max_gap=None):
    color_files = sorted_files(color_dir)
    depth_files = sorted_files(depth_dir)

    c_ts = np.array([parse_last_int_in_name(p.name) for p in color_files], dtype=float)
    d_ts = np.array([parse_last_int_in_name(p.name) for p in depth_files], dtype=float)

    c_idx = np.where(~np.isnan(c_ts))[0]
    d_idx = np.where(~np.isnan(d_ts))[0]
    if len(c_idx) == 0 or len(d_idx) == 0:
        raise RuntimeError("No timestamps found in filenames (expects last integer in stem).")

    order = np.argsort(d_ts[d_idx])
    d_sorted_idx = d_idx[order]
    d_sorted_ts  = d_ts[d_sorted_idx]

    def nearest_depth_index_for_ts(t_c: float):
        pos = bisect_left(d_sorted_ts, t_c)
        cand = []
        if pos < len(d_sorted_ts): cand.append((abs(d_sorted_ts[pos]-t_c),   d_sorted_idx[pos]))
        if pos > 0:                 cand.append((abs(d_sorted_ts[pos-1]-t_c), d_sorted_idx[pos-1]))
        if not cand: return None, np.inf
        gap, didx = min(cand, key=lambda x: x[0])
        return int(didx), float(gap)

    color2depth, depth2colors, pairs = {}, {}, []
    for ic, cf in enumerate(color_files):
        t = c_ts[ic]
        if np.isnan(t): continue
        didx, gap = nearest_depth_index_for_ts(t)
        if didx is None: continue
        if (max_gap is not None) and (gap > max_gap): continue
        c_name, d_name = cf.name, depth_files[didx].name
        color2depth[c_name] = d_name
        depth2colors.setdefault(d_name, []).append(c_name)
        pairs.append((ic, didx, gap))
    return color2depth, depth2colors, pairs

def save_mapping_json(out_path: Path, color_dir: Path, depth_dir: Path, color2depth, depth2colors, max_gap=None):
    payload = {
        "color_dir": str(color_dir),
        "depth_dir": str(depth_dir),
        "timestamp_note": "Mapped by nearest filename-inferred timestamp (last integer in stem).",
        "max_ts_gap": max_gap,
        "color2depth": color2depth,
        "depth2colors": depth2colors,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# ---------- Main ----------
def main():
    color_dir = Path(COLOR_DIR)
    depth_dir = Path(DEPTH_DIR)
    weights   = Path(WEIGHTS)

    if not color_dir.exists(): raise FileNotFoundError(f"COLOR dir not found: {color_dir}")
    if not depth_dir.exists(): raise FileNotFoundError(f"DEPTH dir not found: {depth_dir}")
    if not weights.exists():   raise FileNotFoundError(f"Weights not found: {weights}")

    # Mapping JSON in code/; create if missing
    if not MAPPING_JSON.exists():
        print(f"[info] mapping not found at {MAPPING_JSON}, building…")
        c2d, d2c, pairs = build_timestamp_mapping(color_dir, depth_dir)
        save_mapping_json(MAPPING_JSON, color_dir, depth_dir, c2d, d2c)
        print(f"[info] saved mapping: {MAPPING_JSON}  ({len(c2d)} pairs)")
    else:
        print(f"[info] using mapping: {MAPPING_JSON}")
        with open(MAPPING_JSON, "r", encoding="utf-8") as f:
            _ = json.load(f)  # sanity-check it's readable

    with open(MAPPING_JSON, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    color2depth = mapping.get("color2depth", {})

    print("[info] loading model:", weights)
    model = YOLO(str(weights))

    color_files = sorted_files(color_dir)
    depth_files = sorted_files(depth_dir)
    depth_by_name = {p.name: p for p in depth_files}

    frames_dir = Path(FRAMES_DIR); frames_dir.mkdir(parents=True, exist_ok=True)
    Path(VIDEO_OUT).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    period = 1.0 / max(int(FPS), 1)

    total_keep = total_drop = 0
    print(f"[info] processing {len(color_files)} frames…")

    for i, color_path in enumerate(color_files):
        t0 = time.time()

        rgb = load_color(color_path)
        H, W = rgb.shape[:2]

        depth = None
        dname = color2depth.get(color_path.name)
        if dname and dname in depth_by_name:
            depth = load_depth(depth_by_name[dname])
            if depth.shape[:2] != (H, W):
                depth = None  # require same size

        # YOLO
        r = model.predict(
            source=str(color_path),
            imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU,
            augment=YOLO_AUG, verbose=False
        )[0]
        boxes = r.boxes.xyxy.cpu().numpy().astype(int) if (r.boxes is not None and len(r.boxes)>0) else np.empty((0,4), int)

        if depth is not None:
            kept, dropped = filter_boxes_size_roughness_metric(
                boxes, depth, fx=FX, fy=FY, depth_scale=DEPTH_SCALE,
                min_w_m=MIN_W_M, min_h_m=MIN_H_M, min_aspect=MIN_ASPECT,
                shrink=SHRINK_FRAC, roughness_thr=ROUGH_THR_M
            )
            kept_boxes  = np.array([k["box"] for k in kept], dtype=int) if kept else np.empty((0,4), int)
            kept_labels = [f"h={k['height_m']:.2f}m Z={k['Z_m']:.2f}m r={k['rough_mad_m']*100:.1f}cm" for k in kept]
            drop_boxes  = np.array([b for (b,_r) in dropped], dtype=int) if dropped else np.empty((0,4), int)

            frame = rgb.copy()
            if len(drop_boxes): frame = draw_boxes(frame, drop_boxes, (0,0,255), 1)
            if len(kept_boxes): frame = draw_boxes(frame, kept_boxes, (0,200,0), 2, labels=kept_labels)

            total_keep += len(kept_boxes)
            total_drop += len(drop_boxes)
        else:
            frame = draw_boxes(rgb, boxes, (0,165,255), 2)

        # save frame jpg at full resolution
        out_jpg = Path(FRAMES_DIR) / f"frame_{i:05d}.jpg"
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_jpg), frame)

        # write video
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
            writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, FPS, (w, h))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter failed to open. Try VIDEO_FOURCC='avc1'.")
        writer.write(frame)

        # pacing (optional)
        elapsed = time.time() - t0
        sleep_left = period - elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)

        if (i+1) % 10 == 0 or (i+1) == len(color_files):
            print(f"  {i+1}/{len(color_files)} frames…")

    if writer is not None:
        writer.release()

    print(f"[done] video:  {VIDEO_OUT}")
    print(f"[done] frames: {FRAMES_DIR}/frame_00000.jpg …")
    print(f"[stats] kept={total_keep}  dropped={total_drop}")

if __name__ == "__main__":
    main()
