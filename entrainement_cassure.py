

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping  # <-- Import manquant ajouté
from donnees_cassure import train_generator, val_generator

# Chargement du modèle de base MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Gel des poids

# Architecture personnalisée
x = Dropout(0.5)(base_model.output)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)  # <-- Correction ici
]

# Entraînement
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32,
    epochs=10,
    callbacks=callbacks  # <-- Utilisation correcte
)

# Sauvegarde
model.save("models_cassure/modele_cassure.h5")
print("✅ Modèle sauvegardé sous 'models_cassure/modele_cassure.h5'")