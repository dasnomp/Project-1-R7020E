from ultralytics import YOLO
import cv2
import os



# ===== STEP 1: Camera parameters =====
fx = 306.0002441
fy = 306.112335
cx = 318.475311
cy = 201.369491

# ====== Other parameters ====
DEPTH_SCALE = 0.001               # raw depth units -> meters (0.001 if mm)
MIN_W_M, MIN_H_M = 0.20, 0.70     # meters, used for depth-aware filtration
MIN_ASPECT = 1.20                 # tallness h/w, used for depth-aware filtration
SHRINK_FRAC = 0.08                # inner crop for depth stats, used for depth-aware filtration
ROUGH_THR_M = 0.10                # plane residual MAD (m), used for depth-aware filtration




model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)


# ===== FONCTION 1: DÉTECTION =====
def detecter(image_rgb):
    """
    Détecte les extincteurs avec YOLO
    
    Args:
        image_rgb: Image RGB (numpy array)
    
    Returns:
        Liste de boxes YOLO
    """
    results = model(image_rgb, verbose=False)
    detections = results[0].boxes
    
    
    return detections

# ---- helpers for depth-aware filtration ----
def _shrink_box(x1, y1, x2, y2, W, H, margin=0.08):
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    dx, dy = int(bw*margin), int(bh*margin)
    nx1 = max(0, x1 + dx); ny1 = max(0, y1 + dy)
    nx2 = min(W-1, x2 - dx); ny2 = min(H-1, y2 - dy)
    if nx2 <= nx1 or ny2 <= ny1: return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2

def _plane_residual_mad(depth_patch_m):
    dp = depth_patch_m.astype(np.float32)
    valid = (dp > 0) & np.isfinite(dp)
    if valid.sum() < 200: return np.nan
    H, W = dp.shape
    vv, uu = np.mgrid[0:H, 0:W]
    Z = dp[valid].reshape(-1,1)
    U = uu[valid].reshape(-1,1)
    V = vv[valid].reshape(-1,1)
    X = np.hstack([U, V, np.ones_like(U)])
    coef, *_ = np.linalg.lstsq(X, Z, rcond=None)
    Zhat = (X @ coef).ravel()
    res = np.abs(Z.ravel() - Zhat)
    return float(np.median(np.abs(res - np.median(res))))

def _metric_filter_boxes(
    boxes_xyxy, depth_img_raw, fx, fy, depth_scale,
    min_w_m=0.2, min_h_m=0.7, min_aspect=1.2, shrink=0.08, roughness_thr=0.1
):
    H, W = depth_img_raw.shape[:2]
    kept, dropped = [], []
    for (x1,y1,x2,y2) in boxes_xyxy.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)

        ix1,iy1,ix2,iy2 = _shrink_box(x1,y1,x2,y2,W,H,margin=shrink)
        dep_patch_raw = depth_img_raw[iy1:iy2+1, ix1:ix2+1].astype(np.float32)
        valid = dep_patch_raw[(dep_patch_raw > 0) & np.isfinite(dep_patch_raw)]
        if valid.size < 50: dropped.append(((x1,y1,x2,y2), "no_depth")); continue

        Z = float(np.median(valid)) * depth_scale
        if not np.isfinite(Z) or Z <= 0: dropped.append(((x1,y1,x2,y2), "invalid_Z")); continue

        w_px = x2 - x1 + 1; h_px = y2 - y1 + 1
        width_m  = (w_px * Z) / float(fx)
        height_m = (h_px * Z) / float(fy)
        aspect_m = height_m / max(width_m, 1e-9)

        if width_m < min_w_m:   dropped.append(((x1,y1,x2,y2), f"width<{min_w_m:.2f}m ({width_m:.2f})")); continue
        if height_m < min_h_m:  dropped.append(((x1,y1,x2,y2), f"height<{min_h_m:.2f}m ({height_m:.2f})")); continue
        if aspect_m < min_aspect: dropped.append(((x1,y1,x2,y2), f"aspect<{min_aspect:.2f} ({aspect_m:.2f})")); continue

        rough = _plane_residual_mad(dep_patch_raw * depth_scale)
        if np.isnan(rough):      dropped.append(((x1,y1,x2,y2), "rough_nan")); continue
        if rough < roughness_thr: dropped.append(((x1,y1,x2,y2), f"rough<{roughness_thr:.3f}m ({rough:.3f})")); continue

        kept.append({
            "box": (x1,y1,x2,y2),
            "width_m": width_m, "height_m": height_m, "Z_m": Z,
            "aspect_m": aspect_m, "rough_mad_m": rough
        })
    return kept, dropped

def _detections_to_xyxy(detections):
    if len(detections) == 0:
        return np.empty((0,4), dtype=int)
    out = []
    for b in detections:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
        out.append([int(x1), int(y1), int(x2), int(y2)])
    return np.array(out, dtype=int)

def _draw(img_bgr, boxes_xyxy, color, thickness=2, labels=None):
    out = img_bgr.copy()
    for i,b in enumerate(boxes_xyxy):
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(out, labels[i], (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out

# Depth aware filtration function
def filter_extinguishers_depth(
    detections, rgb_img, rgb_path, output_dir="results",
    depth_folder="datasets/camera_depth_image_raw/camera_depth_image_raw",
    depth_lookup_fn=None  # pass trouver_depth_pour_rgb; if None we'll import lazily
):
    """
    Same interface & returns as your original filter_extinguishers(...),
    but uses depth chosen by `trouver_depth_pour_rgb` (no prebuilt mapping).
    """
    # lazy import to avoid circular import if this lives in functions.py too
    if depth_lookup_fn is None:
        from functions import trouver_depth_pour_rgb
        depth_lookup_fn = trouver_depth_pour_rgb

    rgb_name = os.path.basename(str(rgb_path))

    # 1) Find matching depth path by timestamp
    depth_path = depth_lookup_fn(rgb_name, depth_folder)
    depth = None
    if depth_path and os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None and depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    # 2) If no usable depth, just draw/keep all
    if (depth is None) or (depth.shape[:2] != rgb_img.shape[:2]):
        img_display = rgb_img.copy()
        boxes_xyxy = _detections_to_xyxy(detections)
        labels = [f"EXT {float(b.conf[0]):.2f}" for b in detections]
        if len(boxes_xyxy):
            img_display = _draw(img_display, boxes_xyxy, (0,165,255), 2, labels=labels)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, rgb_name)
        cv2.imwrite(save_path, img_display)
        print(f"    [metric] depth missing/mismatch -> kept all {len(detections)}")
        return list(detections), save_path

    # 3) Metric filter using depth
    boxes_xyxy = _detections_to_xyxy(detections)
    kept, dropped = _metric_filter_boxes(
        boxes_xyxy, depth, fx=fx, fy=fy, depth_scale=DEPTH_SCALE,
        min_w_m=MIN_W_M, min_h_m=MIN_H_M, min_aspect=MIN_ASPECT,
        shrink=SHRINK_FRAC, roughness_thr=ROUGH_THR_M
    )

    # 4) Map back to original YOLO boxes by exact coordinate match
    kept_set = {tuple(k["box"]) for k in kept}
    detections_valides = []
    for b in detections:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        if (x1,y1,x2,y2) in kept_set:
            detections_valides.append(b)

    # 5) Draw and save
    img_display = rgb_img.copy()
    drop_boxes = np.array([d[0] for d in dropped], dtype=int) if dropped else np.empty((0,4), int)
    kept_boxes = np.array([k["box"] for k in kept], dtype=int) if kept else np.empty((0,4), int)
    if len(drop_boxes): img_display = _draw(img_display, drop_boxes, (0,0,255), 1)
    if len(kept_boxes):
        labels = [f"h={k['height_m']:.2f}m Z={k['Z_m']:.2f}m r={k['rough_mad_m']*100:.1f}cm" for k in kept]
        img_display = _draw(img_display, kept_boxes, (0,200,0), 2, labels=labels)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, rgb_name)
    cv2.imwrite(save_path, img_display)

    print(f"    [metric] dropped={len(drop_boxes)} kept={len(kept_boxes)}")
    print(f"    Sauvegardé: {save_path}")
    return detections_valides, save_path





# ===== FONCTION 2: FILTRAGE, filter on paper size ratio =====
def filter_extinguishers(detections, rgb_img, rgb_path, output_dir="results"):
    """
    Filtre les decoys et sauvegarde l'image
    
    Args:
        detections: Détections YOLO brutes
        rgb_img: Image RGB (numpy array)
        rgb_path: Chemin de l'image RGB originale
        output_dir: Dossier de sortie
    
    Returns:
        detections_valides, save_path
    """
    # Dimensions A4/A3
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297
    A3_WIDTH_MM = 297
    A3_HEIGHT_MM = 420
    
    img_display = rgb_img.copy()
    detections_valides = []
    nb_decoys = 0
    
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        
        w = x2 - x1
        h = y2 - y1
        
        # Filtre 1: Confidence et taille
        if confidence < 0.05 or w < 50 or h < 50:
            nb_decoys += 1
            continue
        
        # Filtre 2: Aspect ratio
        aspect_ratio = h / w
        if 1.3 < aspect_ratio < 1.5:
            nb_decoys += 1
            continue
        
        # Filtre 3: Taille A4/A3
        if ((0.9 * A4_WIDTH_MM <= w <= 1.1 * A4_WIDTH_MM and 
             0.9 * A4_HEIGHT_MM <= h <= 1.1 * A4_HEIGHT_MM) or
            (0.9 * A3_WIDTH_MM <= w <= 1.1 * A3_WIDTH_MM and 
             0.9 * A3_HEIGHT_MM <= h <= 1.1 * A3_HEIGHT_MM)):
            nb_decoys += 1
            continue
        
        # C'est un vrai extincteur !
        detections_valides.append(box)
        
        # DESSINER
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_display, f"EXT {confidence:.2f}", (x1, max(0, y1 - 6)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(rgb_path)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img_display)
    
    print(f"    Filtrage: {nb_decoys} decoys, {len(detections_valides)} valides")
    print(f"    Sauvegardé: {save_path}")
    
    return detections_valides, save_path







def trouver_depth_pour_rgb(rgb_image_name, depth_folder):
    """
    Trouve l'image Depth qui correspond à une image RGB
    
    Args:
        rgb_image_name (str): Nom de l'image RGB 
                              Ex: "camera_color_image_1727164479163392418.png"
        depth_folder (str): Chemin vers le dossier des images depth
        
    Returns:
        str: Nom de l'image depth correspondante
             Ex: "camera_depth_image_1727164479142848725.png"
             ou None si pas trouvée
    """
    
    # 1. Extraire le timestamp de l'image RGB
    timestamp_rgb = rgb_image_name.replace('camera_color_image_', '').replace('.png', '')
    
    # 2. Prendre les 10 premiers chiffres (les secondes)
    secondes_rgb = timestamp_rgb[:10]
    
    # 3. Lister toutes les images depth
    depth_images = [f for f in os.listdir(depth_folder) if f.endswith('.png')]
    
    # 4. Chercher les images depth avec les mêmes secondes
    candidates = []
    
    for depth_img in depth_images:
        timestamp_depth = depth_img.replace('camera_depth_image_', '').replace('.png', '')
        secondes_depth = timestamp_depth[:10]
        
        # Si même seconde, c'est un candidat
        if secondes_depth == secondes_rgb:
            # Calculer la différence pour trouver la plus proche
            diff = abs(int(timestamp_rgb) - int(timestamp_depth))
            candidates.append((depth_img, diff))
    
    # 5. Si on a trouvé des candidats, prendre le plus proche
    if candidates:
        candidates.sort(key=lambda x: x[1])
        depth_name = candidates[0][0]
        
        # Retourner le CHEMIN COMPLET
        depth_path = os.path.join(depth_folder, depth_name)
        return depth_path
    
    return None








# ===== FONCTION 3: LOCALISATION 3D =====
def localiser_3d(detections_filtrees, depth_path):
    """
    Calcule la position 3D de chaque extincteur
    
    Args:
        detections_filtrees: Liste de boxes filtrées
        depth_path: CHEMIN COMPLET de l'image depth
    
    Returns:
        Liste de positions 3D
    """
    # 1. Charger l'image depth
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_img is None:
        print(f"   Impossible de charger : {depth_path}")
        return []
    
    print(f"    Depth chargée : {os.path.basename(depth_path)}")
    
    # 2. Calcul 3D
    positions = []
    
    for i, box in enumerate(detections_filtrees):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0])
        
        # Centre du bbox
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2
        
        # Profondeur au centre
        Z_mm = float(depth_img[cy_box, cx_box])
        Z = Z_mm / 1000.0  # mm → m
        
        if Z > 0:
            # Calcul position 3D
            X = (cx_box - cx) * Z / fx
            Y = (cy_box - cy) * Z / fy
            
            positions.append({
                'id': i + 1,
                'bbox': (x1, y1, x2, y2),
                'center': (cx_box, cy_box),
                'X': X,
                'Y': Y,
                'Z': Z,
                'confidence': confidence
            })
    
    print(f"    {len(positions)} extincteurs localisés en 3D")
    
    return positions
