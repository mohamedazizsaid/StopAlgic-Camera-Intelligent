from tensorflow.keras.models import load_model
from donnees_cassure import val_generator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# Chargement du meilleur modèle
model = load_model("models_cassure/best_model.h5")

# Évaluation standard
loss, accuracy, precision, recall = model.evaluate(val_generator)
print(f"\nPerformance sur le jeu de validation:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")

# Rapport complet
y_true = val_generator.classes
y_pred = (model.predict(val_generator) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["defect", "ok"]))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["defect", "ok"])
disp.plot(cmap='Blues')
plt.savefig("models_cassure/confusion_matrix.png")
plt.show()