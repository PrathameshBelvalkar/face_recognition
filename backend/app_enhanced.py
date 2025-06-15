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
import mediapipe as mp
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "known_faces/encodings.pkl"
METADATA_FILE = "known_faces/metadata.json"

# Enhanced configuration
CONFIG = {
    "recognition_threshold": 0.5,  # Lower threshold for better accuracy
    "min_confidence": 0.75,  # Minimum confidence to accept a match
    "max_encodings_per_person": 5,  # Multiple encodings per person
    "face_alignment": True,  # Enable face alignment
    "quality_threshold": 0.7,  # Minimum face quality score
}

# Initialize MediaPipe for face alignment
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Create directory if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


class FaceQualityAnalyzer:
    """Analyze face quality for better recognition"""

    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def calculate_quality_score(self, image):
        """Calculate face quality score based on multiple factors"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)

        if not results.detections:
            return 0.0

        # Get the first detection
        detection = results.detections[0]

        # Quality factors
        confidence = detection.score[0]

        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
        blur_score = min(blur_score, 1.0)

        # Brightness analysis
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2

        # Combined quality score
        quality_score = confidence * 0.4 + blur_score * 0.3 + brightness_score * 0.3

        return min(quality_score, 1.0)


class FaceAligner:
    """Align faces for consistent encoding"""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def align_face(self, image, face_location):
        """Align face based on eye positions"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return image  # Return original if alignment fails

            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]

            # Get eye landmarks (left eye: 33, right eye: 263)
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]

            left_eye_pos = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_pos = (int(right_eye.x * w), int(right_eye.y * h))

            # Calculate angle
            delta_x = right_eye_pos[0] - left_eye_pos[0]
            delta_y = right_eye_pos[1] - left_eye_pos[1]
            angle = np.degrees(np.arctan2(delta_y, delta_x))

            # Rotate image
            center = (
                (left_eye_pos[0] + right_eye_pos[0]) // 2,
                (left_eye_pos[1] + right_eye_pos[1]) // 2,
            )

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h))

            return aligned_image

        except Exception as e:
            logger.warning(f"Face alignment failed: {e}")
            return image


# Initialize quality analyzer and aligner
quality_analyzer = FaceQualityAnalyzer()
face_aligner = FaceAligner()


def load_face_data():
    """Load known faces and metadata"""
    encodings, names = [], []
    metadata = {}

    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                encodings = data.get("encodings", [])
                names = data.get("names", [])
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")

    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")

    return encodings, names, metadata


def save_face_data(encodings, names, metadata):
    """Save known faces and metadata"""
    try:
        # Save encodings
        data = {"encodings": encodings, "names": names}
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)

        # Save metadata
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        logger.error(f"Error saving face data: {e}")
        raise


def base64_to_image(base64_string):
    """Convert base64 image to OpenCV format"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None


def get_face_encoding_with_quality(image, face_location):
    """Get face encoding with quality assessment and alignment"""
    try:
        # Calculate quality score
        quality_score = quality_analyzer.calculate_quality_score(image)

        if quality_score < CONFIG["quality_threshold"]:
            return None, quality_score

        # Align face if enabled
        if CONFIG["face_alignment"]:
            image = face_aligner.align_face(image, face_location)

        # Convert to RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get encoding
        face_encodings = face_recognition.face_encodings(rgb_img, [face_location])

        if face_encodings:
            return face_encodings[0], quality_score
        else:
            return None, quality_score

    except Exception as e:
        logger.error(f"Error getting face encoding: {e}")
        return None, 0.0


def find_best_matches(face_encoding, known_encodings, known_names, metadata):
    """Find best matches with enhanced logic"""
    if not known_encodings:
        return []

    # Calculate distances
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    # Group by person name
    person_distances = defaultdict(list)
    for i, (distance, name) in enumerate(zip(face_distances, known_names)):
        person_distances[name].append((distance, i))

    # Find best match for each person
    matches = []
    for person_name, distances in person_distances.items():
        # Get the best (minimum) distance for this person
        best_distance, best_idx = min(distances, key=lambda x: x[0])

        if best_distance <= CONFIG["recognition_threshold"]:
            confidence = 1 - best_distance
            if confidence >= CONFIG["min_confidence"]:
                matches.append(
                    {
                        "name": person_name,
                        "confidence": round(confidence * 100, 2),
                        "distance": round(best_distance, 3),
                        "encoding_count": len(distances),
                    }
                )

    # Sort by confidence
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    return matches


@app.route("/register", methods=["POST"])
def register_face():
    try:
        data = request.json
        name = data.get("name")
        image_data = data.get("image")
        is_variation = data.get("is_variation", False)  # Allow adding variations

        if not name or not image_data:
            return jsonify({"error": "Name and image are required"}), 400

        # Convert base64 image to OpenCV format
        img = base64_to_image(image_data)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Convert BGR to RGB for face_recognition
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) == 0:
            return jsonify({"error": "No face detected in image"}), 400

        if len(face_locations) > 1:
            return (
                jsonify(
                    {
                        "error": "Multiple faces detected. Please ensure only one face is visible"
                    }
                ),
                400,
            )

        # Get face encoding with quality assessment
        face_encoding, quality_score = get_face_encoding_with_quality(
            img, face_locations[0]
        )

        if face_encoding is None:
            return (
                jsonify(
                    {
                        "error": f"Face quality too low (score: {quality_score:.2f}). Please use a clearer image."
                    }
                ),
                400,
            )

        # Load existing data
        known_encodings, known_names, metadata = load_face_data()

        # Check for existing person
        existing_person_encodings = []
        for i, existing_name in enumerate(known_names):
            if existing_name == name:
                existing_person_encodings.append(known_encodings[i])

        # If not a variation, check if face already exists for someone else
        if not is_variation and known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=CONFIG["recognition_threshold"],
            )
            if any(matches):
                match_index = matches.index(True)
                existing_name = known_names[match_index]
                if existing_name != name:
                    return (
                        jsonify(
                            {
                                "error": f"This face is already registered as {existing_name}"
                            }
                        ),
                        400,
                    )

        # Check if we already have enough encodings for this person
        if len(existing_person_encodings) >= CONFIG["max_encodings_per_person"]:
            return (
                jsonify(
                    {
                        "error": f'Maximum {CONFIG["max_encodings_per_person"]} variations allowed per person'
                    }
                ),
                400,
            )

        # Add new encoding
        known_encodings.append(face_encoding)
        known_names.append(name)

        # Update metadata
        if name not in metadata:
            metadata[name] = {
                "created_at": datetime.now().isoformat(),
                "encoding_count": 0,
                "variations": [],
            }

        metadata[name]["encoding_count"] += 1
        metadata[name]["variations"].append(
            {
                "added_at": datetime.now().isoformat(),
                "quality_score": round(quality_score, 3),
            }
        )

        # Save data
        save_face_data(known_encodings, known_names, metadata)

        # Save image for reference
        variation_suffix = (
            f"_v{len(existing_person_encodings) + 1}"
            if existing_person_encodings
            else ""
        )
        image_path = os.path.join(KNOWN_FACES_DIR, f"{name}{variation_suffix}.jpg")
        cv2.imwrite(image_path, img)

        message = f"{name} registered successfully!"
        if is_variation:
            message += f" (Variation {len(existing_person_encodings) + 1})"

        return (
            jsonify(
                {
                    "message": message,
                    "quality_score": round(quality_score, 3),
                    "total_variations": metadata[name]["encoding_count"],
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/recognize", methods=["POST"])
def recognize_face():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Convert base64 image to OpenCV format
        img = base64_to_image(image_data)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_img)
        if len(face_locations) == 0:
            return (
                jsonify(
                    {"recognized": False, "message": "No face detected", "faces": []}
                ),
                200,
            )

        # Load known faces
        known_encodings, known_names, metadata = load_face_data()

        if len(known_encodings) == 0:
            return (
                jsonify(
                    {
                        "recognized": False,
                        "message": "No registered faces found",
                        "faces": [],
                    }
                ),
                200,
            )

        # Process each detected face
        results = []
        for i, face_location in enumerate(face_locations):
            # Get face encoding with quality assessment
            face_encoding, quality_score = get_face_encoding_with_quality(
                img, face_location
            )

            if face_encoding is None:
                results.append(
                    {
                        "face_id": i + 1,
                        "recognized": False,
                        "message": f"Face quality too low (score: {quality_score:.2f})",
                        "quality_score": round(quality_score, 3),
                    }
                )
                continue

            # Find matches
            matches = find_best_matches(
                face_encoding, known_encodings, known_names, metadata
            )

            if matches:
                best_match = matches[0]
                results.append(
                    {
                        "face_id": i + 1,
                        "recognized": True,
                        "name": best_match["name"],
                        "confidence": best_match["confidence"],
                        "distance": best_match["distance"],
                        "quality_score": round(quality_score, 3),
                        "encoding_variations": best_match["encoding_count"],
                        "all_matches": matches[:3],  # Show top 3 matches
                    }
                )
            else:
                results.append(
                    {
                        "face_id": i + 1,
                        "recognized": False,
                        "message": "Face not recognized",
                        "quality_score": round(quality_score, 3),
                    }
                )

        # Prepare response
        recognized_faces = [r for r in results if r.get("recognized", False)]

        if recognized_faces:
            names = [f["name"] for f in recognized_faces]
            return (
                jsonify(
                    {
                        "recognized": True,
                        "faces": results,
                        "message": f'Recognized: {", ".join(names)}',
                        "total_faces": len(face_locations),
                        "recognized_count": len(recognized_faces),
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "recognized": False,
                        "faces": results,
                        "message": "No faces recognized",
                        "total_faces": len(face_locations),
                    }
                ),
                200,
            )

    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/registered_faces", methods=["GET"])
def get_registered_faces():
    try:
        _, known_names, metadata = load_face_data()

        # Get unique names and their metadata
        unique_names = list(set(known_names))
        face_info = []

        for name in unique_names:
            info = {
                "name": name,
                "encoding_count": known_names.count(name),
                "metadata": metadata.get(name, {}),
            }
            face_info.append(info)

        return (
            jsonify(
                {
                    "faces": face_info,
                    "total_unique_faces": len(unique_names),
                    "total_encodings": len(known_names),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error getting registered faces: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/delete_face/<name>", methods=["DELETE"])
def delete_face(name):
    try:
        known_encodings, known_names, metadata = load_face_data()

        if name not in known_names:
            return jsonify({"error": "Face not found"}), 404

        # Remove all encodings for this person
        indices_to_remove = [i for i, n in enumerate(known_names) if n == name]

        # Remove in reverse order to maintain indices
        for index in sorted(indices_to_remove, reverse=True):
            known_encodings.pop(index)
            known_names.pop(index)

        # Remove from metadata
        if name in metadata:
            del metadata[name]

        # Save updated data
        save_face_data(known_encodings, known_names, metadata)

        # Delete image files
        for i in range(CONFIG["max_encodings_per_person"] + 1):
            suffix = f"_v{i}" if i > 0 else ""
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}{suffix}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)

        return (
            jsonify(
                {
                    "message": f"{name} deleted successfully",
                    "encodings_removed": len(indices_to_remove),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/config", methods=["GET", "POST"])
def handle_config():
    """Get or update configuration"""
    if request.method == "GET":
        return jsonify(CONFIG), 200

    try:
        data = request.json
        for key, value in data.items():
            if key in CONFIG:
                CONFIG[key] = value

        return (
            jsonify(
                {"message": "Configuration updated successfully", "config": CONFIG}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "config": CONFIG,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
