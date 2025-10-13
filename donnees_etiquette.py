import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Création des dossiers de sortie
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configuration
DATA_DIR = "C:/Users/aymen/Desktop/DATA/etiquette"
IMG_SIZE = (224, 224)  # Taille compatible MobileNetV2
BATCH_SIZE = 32

# Augmentation des données pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Chargement des données
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Calcul des poids de classe pour déséquilibre
import numpy as np
class_counts = np.unique(train_generator.classes, return_counts=True)[1]
class_weight = {0: 1., 1: class_counts[0]/class_counts[1]}  # Poids inverse

print(f"\nClasses: {train_generator.class_indices}")
print(f"Class weights: {class_weight}")