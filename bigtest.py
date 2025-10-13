import cv2
import serial
import serial.tools.list_ports
import os
from datetime import datetime
from time import time
import numpy as np
from database import db
from test_bouchon import analyze_bouchon
from test_etiquette import analyze_etiquette
from test_cassure import analyze_cassure

class BouteilleDetector:
    def __init__(self):
        self.responsable_actuel = None
        self.last_anomaly_time = 0
        self.min_anomaly_interval = 1.0  # Minimum 1 second between anomalies
    
    def set_responsable(self, responsable_id):
        """Set the current responsible for detected anomalies"""
        self.responsable_actuel = responsable_id
    
    def preprocess_image(self, image):
        """Preprocess the image to reduce noise and improve detection"""
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        return image
    
    def process_frame(self, image):
        """Process an image and detect anomalies"""
        image = self.preprocess_image(image)
        
        # Detect cassure
        img_cassure, cassure_conclusion = analyze_cassure(image)
        
        current_time = time()
        if cassure_conclusion == "cassure" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
            print(f"Cassure détectée à {datetime.now()}")
            self._ajouter_annotation(image, "CASSURE DETECTEE!", (50, 50), (0, 0, 255))
            db.create_anomalie(
                type_anomalie="cassure",
                description="Cassure détectée sur la bouteille",
                responsable_id=self.responsable_actuel
            )
            self.last_anomaly_time = current_time
            return image, "DEFECT"
        
        # Analyze bouchon and etiquette if no cassure
        h, w, _ = image.shape
        bouchon_crop = image[0:int(0.25*h), :]
        img_bouchon, bouchon_conclusion = analyze_bouchon(bouchon_crop)
        
        etiquette_crop = image[int(0.35*h):int(0.85*h), :]
        img_etiquette, etiquette_conclusion = analyze_etiquette(etiquette_crop)
        
        print(f"Bouchon: {bouchon_conclusion}, Etiquette: {etiquette_conclusion}")
        
        # Draw zones and annotations
        self._dessiner_zone_bouchon(image, bouchon_conclusion, w, h)
        self._dessiner_zone_etiquette(image, etiquette_conclusion, w, h)
        
        # Global conclusion
        conclusion, color = self._get_conclusion_globale(bouchon_conclusion, etiquette_conclusion)
        self._ajouter_annotation(image, f"CONCLUSION: {conclusion}", (50, 50), color)
        
        # Record anomalies if necessary
        signal = "OK"
        if bouchon_conclusion == "defect" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
            db.create_anomalie(
                type_anomalie="bouchon",
                description="Défaut détecté sur le bouchon",
                responsable_id=self.responsable_actuel
            )
            self.last_anomaly_time = current_time
            signal = "DEFECT"
        
        if etiquette_conclusion == "defect" and (current_time - self.last_anomaly_time) > self.min_anomaly_interval:
            db.create_anomalie(
                type_anomalie="etiquette",
                description="Défaut détecté sur l'étiquette",
                responsable_id=self.responsable_actuel
            )
            self.last_anomaly_time = current_time
            signal = "DEFECT"
            
        return image, signal
    
    def _dessiner_zone_bouchon(self, image, conclusion, w, h):
        """Draw the bouchon zone with annotation"""
        reduction = 400
        x1, y1 = reduction, 0
        x2, y2 = w - reduction, int(0.25*h)
        color = (0, 255, 0) if conclusion == "ok" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        self._ajouter_annotation(
            image, 
            f"BOUCHON {conclusion.upper()}", 
            (x1 + 10, int(0.125*h)), 
            color
        )
    
    def _dessiner_zone_etiquette(self, image, conclusion, w, h):
        """Draw the etiquette zone with annotation"""
        reduction = 200
        x1, y1 = reduction, int(0.35*h)
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
        """Determine the global conclusion"""
        if bouchon == "defect" or etiquette == "defect":
            return "DEFECTUEUSE", (0, 0, 255)
        return "OK", (0, 255, 0)
    
    def _ajouter_annotation(self, image, texte, position, color):
        """Add text to the image"""
        cv2.putText(
            image, texte, position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )

def detect_esp32_port():
    """Detect the USB port for ESP32"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'USB' in port.description or 'CH340' in port.description or 'CP210' in port.description:
            return port.device
    return None

def usb_camera_detection(responsable_id=None):
    """Main function for USB camera detection with ESP32"""
    detector = BouteilleDetector()
    if responsable_id:
        detector.set_responsable(responsable_id)
    
    port = detect_esp32_port()
    if port is None:
        print("❌ Aucun port ESP32 détecté. Vérifie la connexion USB.")
        return
    
    print(f"🔌 Connexion au port : {port}")
    try:
        ser = serial.Serial(port, 115200, timeout=5)
    except serial.SerialException as e:
        print(f"Erreur ouverture port série : {e}")
        return
    
    save_folder = os.path.join(os.path.expanduser("~"), "Desktop", "images_esp32")
    os.makedirs(save_folder, exist_ok=True)
    
    buffer = b''
    recording = False
    
    print("📡 Lecture série en cours... Appuyez sur 'q' pour quitter")
    try:
        while True:
            line = ser.readline()
            if b"IMG_START" in line:
                print("📷 Début d'image détectée")
                buffer = b''
                recording = True
                continue
            
            elif b"IMG_END" in line:
                print("✅ Fin d'image détectée")
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_folder, f"img_{now}.jpg")
                with open(filename, "wb") as f:
                    f.write(buffer)
                print(f"💾 Image enregistrée : {filename}")
                recording = False
                
                # Process the image
                image = cv2.imread(filename)
                if image is None:
                    print("❌ Erreur : Impossible de charger l'image")
                    continue
                
                processed_image, signal = detector.process_frame(image)
                ser.write(f"{signal}\n".encode())  # Send signal to ESP32
                
                # Display the processed image
                cv2.imshow('Détection Anomalies Bouteilles', processed_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            if recording:
                buffer += line
            
    except KeyboardInterrupt:
        print("\n⛔ Arrêt manuel.")
    finally:
        ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    usb_camera_detection()