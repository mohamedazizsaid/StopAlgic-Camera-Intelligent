from pymongo import MongoClient
from datetime import datetime
from pymongo.errors import ConnectionFailure
from werkzeug.security import generate_password_hash, check_password_hash

class MongoDB:
    def __init__(self, host='localhost', port=27017, db_name='bouteille_qualite_db'):
        try:
            self.client = MongoClient(f'mongodb://{host}:{port}/', serverSelectionTimeoutMS=5000)
            # Test la connexion immédiatement
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.responsables = self.db['responsables']
            self.anomalies = self.db['anomalies']
            print("Connexion à MongoDB établie avec succès!")
            self.responsables.create_index([("email", 1)], unique=True)

        except ConnectionFailure as e:
            print(f"Échec de la connexion à MongoDB: {e}")
            raise

    def create_responsable(self, nom, prenom, email, password, role="technicien"):
        if self.responsables.find_one({"email": email}):
            raise ValueError("Un responsable avec cet email existe déjà")
            
        responsable_data = {
            "nom": nom,
            "prenom": prenom,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": role,
            "date_creation": datetime.now()
        }
        return self.responsables.insert_one(responsable_data).inserted_id
    
    def authenticate(self, email, password):
        responsable = self.responsables.find_one({"email": email})
        if not responsable:
            return None
        if not check_password_hash(responsable['password_hash'], password):
            return None
        return responsable

    def get_all_responsables(self):
        return list(self.responsables.find({}))
    
    def get_responsable_by_id(self, id):
        return self.responsables.find_one({"_id": id})
    
    def update_responsable(self, id, update_data):
        return self.responsables.update_one({"_id": id}, {"$set": update_data})
    
    def delete_responsable(self, id):
        return self.responsables.delete_one({"_id": id})
    
    # Méthodes pour la collection 'anomalies'
    def create_anomalie(self, type_anomalie, description="", responsable_id=None):
        anomalie_data = {
            "type": type_anomalie,
            "description": description,
            "date_heure": datetime.now(),
            "statut": "non_resolue",
            "responsable_id": responsable_id
        }
        return self.anomalies.insert_one(anomalie_data).inserted_id
    
    def get_all_anomalies(self, filter={}):
        return list(self.anomalies.find(filter))
    
    def get_anomalie_by_id(self, id):
        return self.anomalies.find_one({"_id": id})
    
    def update_anomalie(self, id, update_data):
        return self.anomalies.update_one({"_id": id}, {"$set": update_data})
    
    def delete_anomalie(self, id):
        return self.anomalies.delete_one({"_id": id})
    
    def get_anomalies_by_type(self, type_anomalie):
        return list(self.anomalies.find({"type": type_anomalie}))
    
    def get_anomalies_non_resolues(self):
        return list(self.anomalies.find({"statut": "non_resolue"}))

# Initialisation de la connexion
try:
    db = MongoDB()
except Exception as e:
    print(f"Impossible de se connecter à la base de données: {e}")
    db = None