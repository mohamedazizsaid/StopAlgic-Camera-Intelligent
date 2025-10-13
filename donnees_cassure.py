

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.makedirs("models_cassure", exist_ok=True)

# Configuration identique à l'étiquette
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    "C:/Users/aymen/Desktop/DATA/cassure",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    "C:/Users/aymen/Desktop/DATA/cassure",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("Classes :", train_generator.class_indices)  # Doit afficher {'casse': 0, 'ok': 1} ou inverse