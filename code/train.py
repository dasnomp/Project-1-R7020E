from ultralytics import YOLO

model = YOLO('yolov8n.pt') # load a pretrained model (recommended for training)

# Entraîner sur VOS extincteurs
results = model.train(
    data='datasets/data.yaml',  # Votre fichier config
    epochs=15,                   # 50 passages
    imgsz=640,                   # Taille des images
    batch=16,                    # Images par lot
)

