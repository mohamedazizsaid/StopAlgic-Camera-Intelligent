import cv2
from database import db
from datetime import datetime
from time import time  # Ajout pour gérer le délai entre anomalies
from test_bouchon import analyze_bouchon
from test_etiquette import analyze_etiquette
from test_cassure import analyze_cassure

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
        # Appliquer un flou léger pour réduire le bruit
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Ajuster le contraste
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        return image
    
    def process_frame(self, image):
        """Traite une image et détecte les anomalies"""
        # Prétraitement de l'image
        image = self.preprocess_image(image)
        
        # Détection de cassure
        img_cassure, cassure_conclusion = analyze_cassure(image)
        
        # Vérifier si une cassure est détectée et si le délai minimum est respecté
        current_time = time()
        if cassure_conclusion == "cassure" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
            print(f"Cassure détectée à {datetime.now()}")  # Débogage
            self._ajouter_annotation(image, "CASSURE DETECTEE!", (50, 50), (0, 0, 255))
            db.create_anomalie(
                type_anomalie="cassure",
                description="Cassure détectée sur la bouteille",
                responsable_id=self.responsable_actuel
            )
            self.last_anomaly_time = current_time
            return image
        
        # Détection bouchon et étiquette si pas de cassure
        h, w, _ = image.shape
        
        # Analyse du bouchon (ajustement des proportions pour flux vidéo)
        bouchon_crop = image[0:int(0.25*h), :]  # Réduit à 25% pour mieux cadrer
        img_bouchon, bouchon_conclusion = analyze_bouchon(bouchon_crop)
        
        # Analyse de l'étiquette (ajustement des proportions)
        etiquette_crop = image[int(0.35*h):int(0.85*h), :]  # Ajusté pour flux vidéo
        img_etiquette, etiquette_conclusion = analyze_etiquette(etiquette_crop)
        
        # Débogage : afficher les conclusions
        print(f"Bouchon: {bouchon_conclusion}, Etiquette: {etiquette_conclusion}")
        
        # Dessiner les zones et annotations
        self._dessiner_zone_bouchon(image, bouchon_conclusion, w, h)
        self._dessiner_zone_etiquette(image, etiquette_conclusion, w, h)
        
        # Conclusion globale
        conclusion, color = self._get_conclusion_globale(bouchon_conclusion, etiquette_conclusion)
        self._ajouter_annotation(image, f"CONCLUSION: {conclusion}", (50, 50), color)
        
        # Enregistrer les anomalies si nécessaire, avec délai
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
            
        return image
    
    def _dessiner_zone_bouchon(self, image, conclusion, w, h):
        """Dessine la zone du bouchon avec annotation"""
        reduction = 400
        x1, y1 = reduction, 0
        x2, y2 = w - reduction, int(0.25*h)  # Ajusté pour correspondre à bouchon_crop
        
        color = (0, 255, 0) if conclusion == "ok" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        self._ajouter_annotation(
            image, 
            f"BOUCHON {conclusion.upper()}", 
            (x1 + 10, int(0.125*h)), 
            color
        )
    
    def _dessiner_zone_etiquette(self, image, conclusion, w, h):
        """Dessine la zone de l'étiquette avec annotation"""
        reduction = 200
        x1, y1 = reduction, int(0.35*h)  # Ajusté pour correspondre à etiquette_crop
        x2, y2 = w - reduction, int(0.85*h)
        
        color = (0, 255, 0) if conclusion == "ok" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        self._ajouter_annotation(
            image, 
            f"ETIQUETTE {conclusion.upper()}", 
            (x1 + 10, int(0.6*h)), 
            color
        )
    
    def _get_conclusion_globale(self, bouchon, etiquette):
        """Détermine la conclusion globale"""
        if bouchon == "defect" or etiquette == "defect":
            return "DEFECTUEUSE", (0, 0, 255)
        return "OK", (0, 255, 0)
    
    def _ajouter_annotation(self, image, texte, position, color):
        """Ajoute du texte à l'image"""
        cv2.putText(
            image, texte, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )

def camera_detection(responsable_id=None):
    """Fonction principale pour la détection par caméra"""
    detector = BouteilleDetector()
    if responsable_id:
        detector.set_responsable(responsable_id)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    # Ajuster la résolution de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Détection en cours... Appuyez sur 'q' pour quitter")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de capture vidéo")
                break
            
            processed_frame = detector.process_frame(frame)
            cv2.imshow('Détection Anomalies Bouteilles', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_detection()