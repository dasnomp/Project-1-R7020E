# time.py (version with video recording)
from f import detecter, filter_extinguishers, trouver_depth_pour_rgb, localiser_3d,filter_extinguishers_depth,plot_3d_timeline
import cv2
import os
import time

# Configuration
color_dir = "datasets/camera_color_image_raw/camera_color_image_raw"
depth_dir = "datasets/camera_depth_image_raw/camera_depth_image_raw"

rgb_images = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])

print(f" {len(rgb_images)} images found")

# Calculate dataset FPS
if len(rgb_images) >= 2:
    ts1 = rgb_images[0].replace('camera_color_image_', '').replace('.png', '')
    ts2 = rgb_images[1].replace('camera_color_image_', '').replace('.png', '')
    t1 = float(ts1[:10] + '.' + ts1[10:])
    t2 = float(ts2[:10] + '.' + ts2[10:])
    diff = t2 - t1
    if diff > 0:
        fps_dataset = 1.0 / diff
    else:
        fps_dataset = 10

print(f" Dataset FPS: {fps_dataset:.1f}")

# VIDEO CONFIGURATION
output_video = "results/detection_realtime_s.mp4"
os.makedirs("results", exist_ok=True)

# Read an image to get dimensions
sample_img = cv2.imread(os.path.join(color_dir, rgb_images[0]))
height, width = sample_img.shape[:2]

# Create the VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
video_writer = cv2.VideoWriter(output_video, fourcc, fps_dataset, (width, height))

print(f" Video recording: {output_video}")
print(f" Resolution: {width}x{height}")
print("   Press 'q' to stop\n")

all_data = []


# ===== LOOP =====
for i, rgb_filename in enumerate(rgb_images):
    
    start_time = time.time()
    
    rgb_path = os.path.join(color_dir, rgb_filename)
    rgb_img = cv2.imread(rgb_path)
    
    if rgb_img is None:
        continue
    
    # Pipeline
    detections = detecter(rgb_img)
    #detections_valides, image_filtree_path = filter_extinguishers(detections, rgb_img, rgb_path)
    detections_valides, image_filtree_path = filter_extinguishers_depth(detections, rgb_img, rgb_path, depth_dir) # switch to this for alt. filter
    depth_path = trouver_depth_pour_rgb(os.path.basename(image_filtree_path), depth_dir)
    
    if depth_path:
        positions = localiser_3d(detections_valides, depth_path)
        for pos in positions:
            all_data.append((i, pos['X'], pos['Y'], pos['Z']))
    else:
        positions = []
    
    # Processing FPS
    fps_traitement = 1.0 / (time.time() - start_time)
    
    # Load the filtered image
    img_display = cv2.imread(image_filtree_path)
    
    # Add info on the image
    cv2.putText(img_display, f"FPS: {fps_traitement:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img_display, f"Frame: {i+1}/{len(rgb_images)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_display, f"Detections: {len(positions)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add 3D positions
    for pos in positions:
        x1, y1, x2, y2 = pos['bbox']
        text = f"X:{pos['X']:.2f} Y:{pos['Y']:.2f} Z:{pos['Z']:.2f}"
        cv2.putText(img_display, text, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # WRITE to the video
    video_writer.write(img_display)
    
    # Also display on screen
    cv2.imshow("Real Time (Recording...)", img_display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n Stop requested")
        break
    
    # Progress
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(rgb_images)} images recorded...")

# Release the video
video_writer.release()
cv2.destroyAllWindows()

print(f"\n Video saved: {output_video}")
print(f"   Duration: {len(rgb_images) / fps_dataset:.1f} seconds")

plot_3d_timeline(all_data, 'results/3d_timeline_s.png')
