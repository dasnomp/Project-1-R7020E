from functions import detecter, filter_extinguishers, filter_extinguishers_depth, trouver_depth_pour_rgb, localiser_3d
import cv2
import os


# Chemin de l'image RGB
rgb_path = "C:/Users/rayha/ProjetRL7020E/datasets/camera_color_image_raw/camera_color_image_raw/camera_color_image_1727164479163392418.png"
depth_folder = "camera_depth_image_raw"

# Charger l'image RGB
rgb_img = cv2.imread(rgb_path)

# 1. DÉTECTION
detections = detecter(rgb_img)

# 2. FILTRAGE
# detections_valides, image_filtree_path = filter_extinguishers_depth(detections, rgb_img, rgb_path, depth_folder = depth_folder) # switch to this for alt. filter
detections_valides, image_filtree_path = filter_extinguishers(detections, rgb_img, rgb_path)
image_name = os.path.basename(image_filtree_path)



# 3. TROUVER L'IMAGE DEPTH
depth_path = trouver_depth_pour_rgb(image_name, "datasets/camera_depth_image_raw/camera_depth_image_raw")


if depth_path is None:
    print(" Aucune image depth trouvée !")
else:
    # 4. LOCALISATION 3D
    positions = localiser_3d(detections_valides, depth_path)
    
    # 5. Afficher résultats
    print(f"\n RÉSULTATS FINAUX :")
    for pos in positions:
        print(f"  Extincteur {pos['id']}: X={pos['X']:.3f}m, Y={pos['Y']:.3f}m, Z={pos['Z']:.3f}m")
