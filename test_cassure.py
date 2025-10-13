import tensorflow as tf
from tensorflow import keras


# test_cassure.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model("models_cassure/modele_cassure.h5")

def analyze_cassure(image):
    img_processed = cv2.resize(image, (224, 224)) / 255.0
    img_processed = np.expand_dims(img_processed, axis=0)

    proba = model.predict(img_processed)[0][0]
    label = "ok" if proba > 0.5 else "cassure"

    img_result = image.copy()
    color = (0, 255, 0) if label == "ok" else (0, 0, 255)
    cv2.rectangle(img_result, (10, 10), (img_result.shape[1]-10, img_result.shape[0]-10), color, 2)
    cv2.putText(img_result, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img_result, label

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_path = "C:/Users/arbia/Desktop/datasetttttttttt/test_etiquette_defectt/8d2a5b6e-fee0-486b-9e3c-59416b627e14.jpg"
    img = cv2.imread(img_path)
    res, conclusion = analyze_cassure(img)

    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.title(f"Cassure : {conclusion}")
    plt.axis('off')
    plt.show()
