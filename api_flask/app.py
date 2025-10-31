import os
import json
import uuid
import datetime
from functools import wraps
from bson import ObjectId
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
import jwt
from dotenv import load_dotenv
import ollama

# --- INITIALIZATION ---
load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

CORS(app)
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
ollama_client = ollama.Client(host='http://localhost:11434')

# --- HELPERS ---
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        return super().default(o)

app.json_encoder = JSONEncoder

# --- AUTHENTICATION MIDDLEWARE ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.users.find_one({'_id': ObjectId(data['user_id'])})
            if not current_user:
                return jsonify({'message': 'User not found!'}), 401
            # Make user object available to the route
            g.current_user = current_user
            g.current_user['_id'] = ObjectId(current_user['_id'])
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401
        return f(*args, **kwargs)
    return decorated

# --- AUTH ROUTES ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400
    if mongo.db.users.find_one({'username': username}):
        return jsonify({'message': 'Username already exists'}), 409
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = mongo.db.users.insert_one({
        'username': username,
        'password': hashed_password,
        'createdAt': datetime.datetime.utcnow()
    }).inserted_id
    return jsonify({'message': 'User registered successfully', 'user_id': str(user_id)}), 201

@app.route('/login', methods=['POST'])
def login():
    auth = request.get_json()
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify'}), 401
    user = mongo.db.users.find_one({'username': auth.get('username')})
    if not user:
        return jsonify({'message': 'User not found'}), 401
    if bcrypt.check_password_hash(user['password'], auth.get('password')):
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token, 'username': user['username']})
    return jsonify({'message': 'Invalid password'}), 401

# --- NOTE CRUD ROUTES (PROTECTED) ---
@app.route('/notes', methods=['GET'])
@token_required
def get_all_notes():
    notes_cursor = mongo.db.notes.find({'userId': g.current_user['_id']}).sort('createdAt', -1)
    notes_list = list(notes_cursor)
    # Rename '_id' to 'id' for frontend convenience
    for note in notes_list:
        note['id'] = str(note['_id'])
        del note['_id']
    return jsonify(notes_list)

@app.route('/notes/<note_id>', methods=['GET'])
@token_required
def get_one_note(note_id):
    note = mongo.db.notes.find_one_or_404({'_id': ObjectId(note_id), 'userId': g.current_user['_id']})
    note['id'] = str(note['_id'])
    del note['_id']
    return jsonify(note)

@app.route('/notes', methods=['POST'])
@token_required
def create_note():
    data = request.get_json()
    insert_result = mongo.db.notes.insert_one({
        'title': data['title'],
        'content': data['content'],
        'userId': g.current_user['_id'],
        'createdAt': datetime.datetime.utcnow(),
        'updatedAt': datetime.datetime.utcnow()
    })
    new_note = mongo.db.notes.find_one({'_id': insert_result.inserted_id})
    new_note['id'] = str(new_note['_id'])
    del new_note['_id']
    regenerate_graph_cache_for_user(g.current_user['_id'])
    return jsonify(new_note), 201

@app.route('/notes/<note_id>', methods=['PUT'])
@token_required
def update_note(note_id):
    data = request.get_json()
    update_data = {
        'title': data['title'],
        'content': data['content'],
        'updatedAt': datetime.datetime.utcnow()
    }
    mongo.db.notes.update_one(
        {'_id': ObjectId(note_id), 'userId': g.current_user['_id']},
        {'$set': update_data}
    )
    updated_note = mongo.db.notes.find_one({'_id': ObjectId(note_id)})
    updated_note['id'] = str(updated_note['_id'])
    del updated_note['_id']
    regenerate_graph_cache_for_user(g.current_user['_id'])
    return jsonify(updated_note)

@app.route('/notes/<note_id>', methods=['DELETE'])
@token_required
def delete_note(note_id):
    result = mongo.db.notes.delete_one({'_id': ObjectId(note_id), 'userId': g.current_user['_id']})
    if result.deleted_count == 1:
        regenerate_graph_cache_for_user(g.current_user['_id'])
        return jsonify({}), 204
    else:
        return jsonify({'message': 'Note not found or permission denied'}), 404

# --- GRAPH AI LOGIC (PROTECTED & USER-SCOPED) ---
def regenerate_graph_cache_for_user(user_id):
    print(f"BACKGROUND: Starting graph regeneration for user {user_id}...")
    try:
        notes = list(mongo.db.notes.find({'userId': user_id}))
        user_id_str = str(user_id)
        cache_file = os.path.join(os.path.dirname(__file__), f'cache_{user_id_str}.json')
        
        if not notes:
            with open(cache_file, 'w') as f: json.dump({"nodes": [], "edges": []}, f, indent=2)
            print(f"BACKGROUND: No notes for user {user_id_str}. Cache cleared.")
            return

        all_content_for_ai = "\n\n".join([f"--- Note ID: {note['_id']} ---\nTitle: {note['title']}\nContent: {note['content']}" for note in notes])
        
        prompt = f"""
        Analyze the provided text carefully. Identify key entities (people, projects, organizations, locations, concepts) and their relationships.
        IMPORTANT RULES:
        1. Each node MUST have: "id" (unique string), "label" (display name), "sourceNoteId" (which note it came from), and "group" (classification)
        2. Valid groups: "Person", "Project", "Organization", "Location", "Concept"
        3. Each edge MUST have: "from" (node id), "to" (node id), "label" (relationship description), "sourceNoteId"
        4. Extract meaningful relationships, not just word co-occurrence.
        5. Node IDs should be descriptive and lowercase (e.g., "person_john_doe").
        Respond ONLY with a valid JSON object with this exact structure:
        {{
          "nodes": [{{"id": "string", "label": "string", "sourceNoteId": "string", "group": "Person|Project|Organization|Location|Concept"}}],
          "edges": [{{"from": "string", "to": "string", "label": "string", "sourceNoteId": "string"}}]
        }}
        Text to analyze:
        {all_content_for_ai}
        """
        
        print(f"BACKGROUND: Calling Ollama for user {user_id_str}...")
        response = ollama_client.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}], format='json', stream=False)
        
        graph_data = json.loads(response['message']['content'])
        
        # Data Validation & Cleaning
        if 'nodes' not in graph_data: graph_data['nodes'] = []
        if 'edges' not in graph_data: graph_data['edges'] = []
        
        node_ids = {node.get('id') for node in graph_data['nodes']}
        graph_data['edges'] = [edge for edge in graph_data['edges'] if edge.get('from') in node_ids and edge.get('to') in node_ids]
        
        with open(cache_file, 'w') as f: json.dump(graph_data, f, indent=2)
        print(f"✅ BACKGROUND: Graph regeneration complete for user {user_id_str}.")

    except Exception as e:
        print(f"❌ BACKGROUND ERROR for user {user_id}: {e}")

@app.route('/graph', methods=['GET'])
@token_required
def get_graph():
    user_id_str = str(g.current_user['_id'])
    cache_file = os.path.join(os.path.dirname(__file__), f'cache_{user_id_str}.json')
    if not os.path.exists(cache_file):
        regenerate_graph_cache_for_user(g.current_user['_id'])
    
    with open(cache_file, 'r') as f:
        return jsonify(json.load(f))

@app.route('/graph/regenerate', methods=['POST'])
@token_required
def force_regenerate():
    # We trigger it but don't wait for it to finish
    regenerate_graph_cache_for_user(g.current_user['_id'])
    return jsonify({'message': 'Graph regeneration triggered'}), 202

# --- SERVER STARTUP ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)