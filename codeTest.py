from ultralytics import YOLO
import cv2

# Charger le modèle entraîné
model = YOLO("C:/Users/Ali/Desktop/ProjetStage/best (4).pt")

# Charger une image depuis ton PC
image_path = r"C:\Users\Ali\Desktop\image pfa\7.jpg"  # chemin corrigé
image = cv2.imread(image_path)

if image is None:
    print("Erreur : Impossible de charger l'image.")
    exit()

# Inférence YOLO
results = model(image, conf=0.5)

# Annoter l'image avec les résultats
annotated_image = results[0].plot()

# Afficher le résultat
cv2.imshow("YOLOv8 - Image Test", annotated_image)
cv2.waitKey(0)  # attend une touche
cv2.destroyAllWindows()
