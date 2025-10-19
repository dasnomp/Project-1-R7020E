from functions import detecter, filter_extinguishers, trouver_depth_pour_rgb, localiser_3d
import cv2
import os

# ===== CONFIGURATION =====
color_dir = "datasets/camera_color_image_raw/camera_color_image_raw"
depth_dir = "datasets/camera_depth_image_raw/camera_depth_image_raw"

# Lister toutes les images RGB
rgb_images = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])

print(f"📂 {len(rgb_images)} images trouvées")
print("🎬 Démarrage simulation temps réel...")
print("   Appuyez sur 'q' pour quitter\n")

# ===== BOUCLE TEMPS RÉEL =====
for i, rgb_filename in enumerate(rgb_images):
    
    print(f"[Image {i+1}/{len(rgb_images)}] {rgb_filename}")
    
    # Construire le chemin complet
    rgb_path = os.path.join(color_dir, rgb_filename)
    
    # Charger l'image
    rgb_img = cv2.imread(rgb_path)
    
    if rgb_img is None:
        print("   ⚠️ Impossible de charger, skip")
        continue
    
    # Pipeline complet
    detections = detecter(rgb_img)
    detections_valides, image_filtree_path = filter_extinguishers(detections, rgb_img, rgb_path)
    
    depth_path = trouver_depth_pour_rgb(os.path.basename(image_filtree_path), depth_dir)
    
    if depth_path:
        positions = localiser_3d(detections_valides, depth_path)
        
        # Afficher résultats
        for pos in positions:
            print(f"   🎯 X={pos['X']:.2f}m Y={pos['Y']:.2f}m Z={pos['Z']:.2f}m")
    
    # Charger l'image filtrée
    img_display = cv2.imread(image_filtree_path)
    
    # AFFICHER
    cv2.imshow("Temps Reel", img_display)
    
    # Attendre 100ms ou 'q' pour quitter
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("\n⏹️  Arrêt")
        break

cv2.destroyAllWindows()
print("\n✅ Terminé !")