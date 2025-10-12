#from ultralytics import YOLO

# Charger VOTRE modèle entraîné
#model = YOLO('runs/detect/train/weights/best.pt')

# Tester sur une image de validation
#results = model('datasets/valid/images')  # Teste sur toutes les images valid

# Afficher la première détection
#results[1].show()


from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# Tester ET sauvegarder
results = model.predict(
    source='datasets/valid/images',
    save=True,   
    project='runs/detect',
    name='predict'
)
