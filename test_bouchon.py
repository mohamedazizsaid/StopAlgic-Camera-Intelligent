
import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

MODEL_PATH = "runs/detect/bouchon/weights/best.pt"
CLASS_NAMES = {0: "defect", 1: "ok"}
model = YOLO(MODEL_PATH)

def analyze_bouchon(image):
    h, w, _ = image.shape
    if w / h > 3:  # si très étirée, recadrer
        start_x = int(0.2 * w)
        end_x = int(0.8 * w)
        image = image[:, start_x:end_x]

    results = model(image, conf=0.3)
    img_result = image.copy()
    counts = {0: 0, 1: 0}

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            counts[cls] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{CLASS_NAMES[cls]} ({conf*100:.1f}%)"
            color = (0, 255, 0) if cls == 1 else (0, 0, 255)
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_result, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    total = sum(counts.values())
    if total == 0:
        conclusion = "Pas de bouchon détecté"
    else:
        defect_percent = counts[0] / total * 100
        ok_percent = counts[1] / total * 100
        conclusion = "defect" if defect_percent > 50 else "ok"
    
    return img_result, conclusion

if __name__ == "__main__":
    img_path = "C:/Users/aymen/Pictures/Screenshots/bouchon (1).png"
    img = cv2.imread(img_path)
    res, conclusion = analyze_bouchon(img)

    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title(f"Bouchon : {conclusion}")
    plt.axis('off')
    plt.show()
