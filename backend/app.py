from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
import os
import json
import pickle
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "known_faces/encodings.pkl"

# Create directory if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces from file
def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    return [], []

# Save known faces to file
def save_known_faces(encodings, names):
    data = {'encodings': encodings, 'names': names}
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(data, f)

# Convert base64 image to OpenCV format
def base64_to_image(base64_string):
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

@app.route('/register', methods=['POST'])
def register_face():
    try:
        data = request.json
        name = data.get('name')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'error': 'Name and image are required'}), 400
        
        # Convert base64 image to OpenCV format
        img = base64_to_image(image_data)
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) == 0:
            return jsonify({'error': 'No face detected in image'}), 400
        
        if len(face_locations) > 1:
            return jsonify({'error': 'Multiple faces detected. Please ensure only one face is visible'}), 400
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        face_encoding = face_encodings[0]
        
        # Load existing encodings
        known_encodings, known_names = load_known_faces()
        
        # Check if person already exists
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            if any(matches):
                match_index = matches.index(True)
                existing_name = known_names[match_index]
                return jsonify({'error': f'This face is already registered as {existing_name}'}), 400
        
        # Add new face
        known_encodings.append(face_encoding)
        known_names.append(name)
        
        # Save to file
        save_known_faces(known_encodings, known_names)
        
        # Save image for reference
        image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        cv2.imwrite(image_path, img)
        
        return jsonify({'message': f'{name} registered successfully!'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert base64 image to OpenCV format
        img = base64_to_image(image_data)
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) == 0:
            return jsonify({'recognized': False, 'message': 'No face detected'}), 200
        
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        # Load known faces
        known_encodings, known_names = load_known_faces()
        
        if len(known_encodings) == 0:
            return jsonify({'recognized': False, 'message': 'No registered faces found'}), 200
        
        # Check each detected face
        results = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            if any(matches):
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    results.append({
                        'name': name,
                        'confidence': round(confidence * 100, 2)
                    })
        
        if results:
            return jsonify({
                'recognized': True,
                'faces': results,
                'message': f'Recognized: {", ".join([f["name"] for f in results])}'
            }), 200
        else:
            return jsonify({'recognized': False, 'message': 'Face not recognized'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/registered_faces', methods=['GET'])
def get_registered_faces():
    try:
        _, known_names = load_known_faces()
        return jsonify({'names': known_names, 'count': len(known_names)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_face/<name>', methods=['DELETE'])
def delete_face(name):
    try:
        known_encodings, known_names = load_known_faces()
        
        if name not in known_names:
            return jsonify({'error': 'Face not found'}), 404
        
        # Remove from lists
        index = known_names.index(name)
        known_encodings.pop(index)
        known_names.pop(index)
        
        # Save updated data
        save_known_faces(known_encodings, known_names)
        
        # Delete image file
        image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return jsonify({'message': f'{name} deleted successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)