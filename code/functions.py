from ultralytics import YOLO
import cv2
import os



# ===== STEP 1: Camera parameters =====
fx = 306.0002441
fy = 306.112335
cx = 318.475311
cy = 201.369491



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






# ===== FONCTION 2: FILTRAGE =====
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