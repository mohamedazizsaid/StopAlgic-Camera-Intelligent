import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_PATH = "runs/detect/bouchon_20250422_030608/weights/best.pt"
CLASS_NAMES = {0: "bouchon_defect", 1: "bouchon_ok"}

# Charger une seule fois le modèle
model = YOLO(MODEL_PATH)

def analyze_and_plot(image_path):
    # Lire l'image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Erreur : Image non chargée")
        return

    # Prédiction
    results = model(img, conf=0.5)

    # Initialiser les compteurs
    counts = {0: 0, 1: 0}

    # Dessiner les boxes
    img_result = img.copy()
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            counts[cls] += 1

            # Dessiner rectangle + label + score
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{CLASS_NAMES[cls]} ({conf*100:.1f}%)"
            color = (0, 255, 0) if cls == 1 else (0, 0, 255)
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_result, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calcul total et pourcentages
    total = sum(counts.values())
    if total == 0:
        print("⚠️ Aucun bouchon détecté")
        return

    defect_percent = counts[0] / total * 100
    ok_percent = counts[1] / total * 100

    # Affichage Console
    print("\n===== 📊 RÉSULTATS ANALYSE BOUCHON =====")
    print(f"- 📌 Défectueux : {counts[0]} ({defect_percent:.1f}%)")
    print(f"- ✅ OK         : {counts[1]} ({ok_percent:.1f}%)")

    # Conclusion
    conclusion = "DÉFECTUEUX" if defect_percent > 50 else "OK"
    print(f"\n➡️  ✅ Conclusion : Bouchon {conclusion}")
    print("========================================\n")

    # Affichage des images côte à côte
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Image Originale")
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Analyse et Détection")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "C:/Users/aymen/Pictures/Screenshots/bouchon (1).png"
    if os.path.exists(image_path):
        analyze_and_plot(image_path)
    else:
        print("❌ Image introuvable :", image_path)
