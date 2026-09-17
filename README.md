# StopAlgic Camera Intelligent

Système de vision industrielle pour le contrôle qualité de bouteilles en production.
Le projet détecte automatiquement trois types d’anomalies :
- **cassure** de bouteille,
- **défaut de bouchon**,
- **défaut d’étiquette**.

Les analyses s’appuient sur des modèles **TensorFlow/Keras** (cassure, étiquette) et **YOLO Ultralytics** (bouchon), avec enregistrement des anomalies dans **MongoDB**.

## Sommaire
- [Architecture du projet](#architecture-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [API d’authentification et anomalies](#api-dauthentification-et-anomalies)
- [Entraînement et évaluation](#entraînement-et-évaluation)
- [Structure des fichiers](#structure-des-fichiers)
- [Améliorations recommandées](#améliorations-recommandées)

## Architecture du projet

### Chaîne de détection
1. Acquisition d’image (caméra USB ou flux ESP32).
2. Prétraitement image (réduction bruit + contraste).
3. Détection prioritaire de **cassure**.
4. Si pas de cassure : analyse **bouchon** + **étiquette**.
5. Décision globale (**OK** / **DEFECTUEUSE**) + annotation visuelle.
6. Journalisation des anomalies dans MongoDB.

### Composants principaux
- **Détection temps réel USB/ESP32** : `bigtest.py`
- **Détection caméra locale** : `prediction.py`
- **Détection image statique** : `static_pred.py`
- **Service API Flask (auth + anomalies)** : `auth.py`
- **Accès MongoDB** : `database.py`

## Prérequis
- Python **3.10+**
- MongoDB (local ou distant)
- Caméra USB (optionnel) / ESP32-CAM (optionnel)

## Installation
```bash
# 1) Cloner le dépôt
# 2) Se placer dans le dossier du projet
cd /home/runner/work/StopAlgic-Camera-Intelligent/StopAlgic-Camera-Intelligent

# 3) Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 4) Installer les dépendances
pip install --upgrade pip
pip install \
  flask flask-cors pyjwt pymongo werkzeug bson \
  opencv-python numpy matplotlib pyserial \
  tensorflow ultralytics scikit-learn
```

## Configuration

### 1) Base de données MongoDB
Par défaut, `database.py` tente une connexion sur :
- `mongodb://localhost:27017/`
- base : `bouteille_qualite_db`

Lancer MongoDB avant d’exécuter les scripts de détection/API.

### 2) Clé secrète JWT (API)
Définir une variable d’environnement pour la clé JWT :
```bash
export SECRET_KEY="votre_cle_secrete"
```

### 3) Modèles attendus
- `models/best_model.h5` (étiquette)
- `models_cassure/modele_cassure.h5` (cassure)
- `runs/detect/bouchon/weights/best.pt` (bouchon)

## Utilisation

### Détection via caméra locale
```bash
python prediction.py
```

### Détection via ESP32-CAM (série)
```bash
python bigtest.py
```
Le script détecte automatiquement un port USB type CH340/CP210.

### Analyse d’image statique
```bash
python static_pred.py
```

## API d’authentification et anomalies
Lancer l’API Flask :
```bash
python auth.py
```
API disponible sur `http://0.0.0.0:5000`.

Endpoints principaux :
- `POST /register` : création utilisateur
- `POST /login` : authentification JWT
- `GET /anomalies` : liste des anomalies
- `GET /anomalies/<id>` : détail anomalie
- `DELETE /anomalies/<id>` : suppression anomalie

## Entraînement et évaluation

### Étiquette
```bash
python entrainement_etiquette.py
python evaluation-etiquette.py
```

### Cassure
```bash
python entrainement_cassure.py
python evaluation_cassure.py
```

### Bouchon (YOLO)
```bash
python train_bouchon_continue.py
```

> **Note :** plusieurs scripts de données (`donnees_etiquette.py`, `donnees_cassure.py`) contiennent des chemins locaux Windows à adapter à votre environnement.

## Structure des fichiers
- `auth.py` : API Flask (authentification + endpoints anomalies)
- `database.py` : couche d’accès MongoDB
- `prediction.py` : détection en direct caméra USB
- `bigtest.py` : pipeline ESP32-CAM + détection
- `static_pred.py` : inférence sur image statique
- `test_bouchon.py` : inférence YOLO bouchon
- `test_etiquette.py` : inférence modèle étiquette
- `test_cassure.py` : inférence modèle cassure
- `entrainement_*.py` / `evaluation_*.py` : entraînement & métriques

## Améliorations recommandées
- Centraliser les dépendances dans un `requirements.txt`.
- Externaliser tous les chemins datasets/modèles dans un fichier `.env` ou YAML.
- Ajouter des tests unitaires et tests d’intégration.
- Ajouter une CI (lint + tests + vérification modèles).
