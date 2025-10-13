import cv2
import matplotlib.pyplot as plt
from test_bouchon import analyze_bouchon
from test_etiquette import analyze_etiquette
from test_cassure import analyze_cassure
from database import db
from datetime import datetime
from time import time

class BouteilleDetector:
    def __init__(self):
        self.responsable_actuel = None
        self.last_anomaly_time = 0
        self.min_anomaly_interval = 1.0  # Délai minimum de 1 seconde entre anomalies
    
    def set_responsable(self, responsable_id):
        """Définir le responsable actuel pour les anomalies détectées"""
        self.responsable_actuel = responsable_id
    
    def preprocess_image(self, image):
        """Prétraitement de l'image pour réduire le bruit et améliorer la détection"""
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        return image

    def prediction_bouteille(self, image_path, responsable_id=None):
        """Analyse une image statique pour détecter les anomalies sur une bouteille"""
        if responsable_id:
            self.set_responsable(responsable_id)
        
        image = cv2.imread(image_path)
        if image is None:
            print("Erreur : Impossible de charger l'image")
            return

        # Prétraitement de l'image
        image = self.preprocess_image(image)
        
        img_cassure, cassure_conclusion = analyze_cassure(image)
        current_time = time()

        if cassure_conclusion == "cassure" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
            print("Cassure détectée !")
            print("Conclusion : La bouteille est cassée.")
            db.create_anomalie(
                type_anomalie="cassure",
                description="Cassure détectée sur la bouteille",
                responsable_id=self.responsable_actuel
            )
            self.last_anomaly_time = current_time
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Image Originale")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img_cassure, cv2.COLOR_BGR2RGB))
            plt.title("Détection Cassure")
            plt.axis('off')
            plt.show()
            return

        else:
            print("Pas de cassure détectée, analyse bouchon et étiquette...")

            h, w, _ = image.shape
            bouchon_crop = image[0:int(0.3*h), :]
            etiquette_crop = image[int(0.4*h):int(0.9*h), :]

            img_bouchon, bouchon_conclusion = analyze_bouchon(bouchon_crop)
            img_etiquette, etiquette_conclusion = analyze_etiquette(etiquette_crop)

            img_result = image.copy()

            # Réduction pour BOUCHON
            reduction_bouchon = 400  
            x1_bouchon = reduction_bouchon
            y1_bouchon = 0
            x2_bouchon = w - reduction_bouchon
            y2_bouchon = int(0.3*h)

            # Réduction pour ETIQUETTE
            reduction_etiquette = 200
            x1_etiquette = reduction_etiquette
            y1_etiquette = int(0.4*h)
            x2_etiquette = w - reduction_etiquette
            y2_etiquette = int(0.8*h)

            # --- Bouchon ---
            color_bouchon = (0, 255, 0) if bouchon_conclusion == "ok" else (0, 0, 255)
            cv2.rectangle(img_result, (x1_bouchon, y1_bouchon), (x2_bouchon, y2_bouchon), color_bouchon, 5)
            cv2.putText(
                img_result,
                f"BOUCHON {bouchon_conclusion.upper()}",
                (x1_bouchon + 10, int(0.15*h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                color_bouchon,
                4,
                cv2.LINE_AA
            )

            # --- Etiquette ---
            color_etiquette = (0, 255, 0) if etiquette_conclusion == "ok" else (0, 0, 255)
            cv2.rectangle(img_result, (x1_etiquette, y1_etiquette), (x2_etiquette, y2_etiquette), color_etiquette, 5)
            cv2.putText(
                img_result,
                f"ETIQUETTE {etiquette_conclusion.upper()}",
                (x1_etiquette + 10, int(0.6*h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                color_etiquette,
                4,
                cv2.LINE_AA
            )

            # Enregistrer les anomalies si nécessaire
            if bouchon_conclusion == "defect" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
                db.create_anomalie(
                    type_anomalie="bouchon",
                    description="Défaut détecté sur le bouchon",
                    responsable_id=self.responsable_actuel
                )
                self.last_anomaly_time = current_time
            
            if etiquette_conclusion == "defect" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
                db.create_anomalie(
                    type_anomalie="etiquette",
                    description="Défaut détecté sur l'étiquette",
                    responsable_id=self.responsable_actuel
                )
                self.last_anomaly_time = current_time

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Image Originale")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
            plt.title("Détection Bouchon / Etiquette")
            plt.axis('off')
            plt.show()

            print(f"Bouchon : {bouchon_conclusion.upper()}")
            print(f"Etiquette : {etiquette_conclusion.upper()}")

            if bouchon_conclusion == "defect" or etiquette_conclusion == "defect":
                print("Conclusion : BOUTEILLE DEFECTUEUSE")
            else:
                print("Conclusion : BOUTEILLE OK")

if __name__ == "__main__":
    detector = BouteilleDetector()
    img_path = "C:/Users/user/Desktop/aaaa.jpg"
    detector.prediction_bouteille(img_path)