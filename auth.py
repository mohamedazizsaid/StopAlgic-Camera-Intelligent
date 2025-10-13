from flask import Flask, request, jsonify
from functools import wraps
import jwt
from datetime import datetime, timedelta
from database import MongoDB
import os
import secrets
from flask_cors import CORS
from bson import ObjectId

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "http://192.168.1.30:5000", "*"]}})
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or secrets.token_urlsafe(32)
db = MongoDB()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
            
        if not token:
            return jsonify({'message': 'Token manquant!'}), 401
            
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            # Vérifier seulement que le token est valide, sans passer current_user
            if not db.get_responsable_by_id(ObjectId(data['user_id'])):
                return jsonify({'message': 'Utilisateur non trouvé!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token invalide!'}), 401
        except Exception as e:
            return jsonify({'message': f'Erreur de token: {str(e)}'}), 401
            
        return f(*args, **kwargs)
        
    return decorated

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not all(key in data for key in ['nom', 'prenom', 'email', 'password']):
        return jsonify({'message': 'Données incomplètes!'}), 400
    
    try:
        user_id = db.create_responsable(
            nom=data['nom'],
            prenom=data['prenom'],
            email=data['email'],
            password=data['password'],
            role=data.get('role', 'technicien')
        )
        return jsonify({'message': 'Utilisateur créé avec succès!', 'user_id': str(user_id)}), 201
    except ValueError as e:
        return jsonify({'message': str(e)}), 400

@app.route('/login', methods=['POST'])
def login():
    auth = request.get_json()
    
    if not auth or not auth.get('email') or not auth.get('password'):
        return jsonify({'message': 'Email ou mot de passe manquant!'}), 401
    
    user = db.authenticate(auth['email'], auth['password'])
    
    if not user:
        return jsonify({'message': 'Identifiants incorrects!'}), 401
    
    token = jwt.encode({
        'user_id': str(user['_id']),
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({'token': token, 'email': user['email']})

@app.route('/protected', methods=['GET'])
def protected_route():
    # Maintenant vous pouvez obtenir l'email depuis les préférences côté client
    return jsonify({'message': 'Route protégée accessible!'})

@app.route('/anomalies', methods=['GET'])
def get_all_anomalies():
    try:
        anomalies = db.get_all_anomalies()
        anomalies_list = []
        for anomalie in anomalies:
            anomalie['_id'] = str(anomalie['_id'])
            if 'responsable_id' in anomalie and anomalie['responsable_id']:
                anomalie['responsable_id'] = str(anomalie['responsable_id'])
            anomalies_list.append(anomalie)
        return jsonify(anomalies_list), 200
    except Exception as e:
        return jsonify({'message': f'Erreur lors de la récupération des anomalies: {str(e)}'}), 500

@app.route('/anomalies/<anomalie_id>', methods=['GET'])
def get_anomalie_by_id(anomalie_id):
    try:
        anomalie = db.get_anomalie_by_id(ObjectId(anomalie_id))
        if not anomalie:
            return jsonify({'message': 'Anomalie non trouvée!'}), 404
        
        anomalie['_id'] = str(anomalie['_id'])
        if 'responsable_id' in anomalie and anomalie['responsable_id']:
            anomalie['responsable_id'] = str(anomalie['responsable_id'])
        return jsonify(anomalie), 200
    except Exception as e:
        return jsonify({'message': f'Erreur lors de la récupération de l\'anomalie: {str(e)}'}), 500

@app.route('/anomalies/<anomalie_id>', methods=['DELETE'])
def delete_anomalie(anomalie_id):
    try:
        result = db.delete_anomalie(ObjectId(anomalie_id))
        if result.deleted_count == 0:
            return jsonify({'message': 'Anomalie non trouvée!'}), 404
        return jsonify({'message': 'Anomalie supprimée avec succès!'}), 200
    except Exception as e:
        return jsonify({'message': f'Erreur lors de la suppression de l\'anomalie: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)