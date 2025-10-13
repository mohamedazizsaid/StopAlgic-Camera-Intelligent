from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)
from donnees_etiquette import train_generator, val_generator, class_weight
import datetime

# --- Configuration ---
BATCH_SIZE = 32  # Ajouté ici pour corriger l'erreur
IMG_SIZE = (224, 224)

# --- Modèle ---
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Gel des couches

x = Dropout(0.5)(base_model.output)
x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- Compilation ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
)

# --- Callbacks ---
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3),
    ModelCheckpoint("models/best_model.h5", save_best_only=True),
    TensorBoard(log_dir=log_dir)
]

# --- Entraînement ---
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,  # Utilise BATCH_SIZE
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=50,
    class_weight=class_weight,
    callbacks=callbacks
)

model.save("models/final_model.h5")