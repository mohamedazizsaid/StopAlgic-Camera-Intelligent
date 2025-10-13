import tensorflow as tf
from tensorflow import keras


import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model("models/best_model.h5")

def analyze_etiquette(image):
    def detect_etiquette_presence(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        non_zero = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        ratio = non_zero / total_pixels
        return ratio > 0.05

    if not detect_etiquette_presence(image):
        return image, "defect"

    img_processed = cv2.resize(image, (224, 224)) / 255.0
    img_processed = np.expand_dims(img_processed, axis=0)

    proba = model.predict(img_processed)[0][0]
    label = "ok" if proba > 0.5 else "defect"

    img_result = image.copy()
    color = (0, 255, 0) if label == "ok" else (0, 0, 255)
    cv2.rectangle(img_result, (10, 10), (img_result.shape[1]-10, img_result.shape[0]-10), color, 2)
    cv2.putText(img_result, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img_result, label

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_path = "C:/Users/aymen/Pictures/Screenshots/t (3).png"
    img = cv2.imread(img_path)
    res, conclusion = analyze_etiquette(img)

    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title(f"Etiquette : {conclusion}")
    plt.axis('off')
    plt.show()
