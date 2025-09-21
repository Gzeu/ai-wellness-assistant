"""AI Wellness Assistant - MVP Version
Simplified version with core functionality for demonstration and testing
Author: Pricop George
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import json
import logging
from datetime import datetime
import threading
import time
from pathlib import Path

# Import configuration
from config import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
CORS(app)

class SimpleStressDetector:
    """Simplified stress detector for MVP"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.stress_history = []
        self.current_stress_level = 0.0
        
    def detect_faces(self, frame):
        """Basic face detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=MIN_FACE_SIZE
        )
        return faces
    
    def calculate_basic_stress(self, face_roi):
        """Calculate basic stress level from face ROI"""
        if face_roi is None or face_roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Calculate basic features
        # Eye region analysis (upper 40% of face)
        h, w = gray.shape
        eye_region = gray[0:int(h*0.4), :]
        
        # Mouth region analysis (lower 30% of face)
        mouth_region = gray[int(h*0.7):, :]
        
        # Calculate intensity variations (tension indicators)
        eye_std = np.std(eye_region) if eye_region.size > 0 else 0
        mouth_std = np.std(mouth_region) if mouth_region.size > 0 else 0
        
        # Combine features for stress score
        stress_score = min(1.0, (eye_std + mouth_std) / 100.0)
        
        # Add some randomization for demo purposes (remove in production)
        base_variation = np.random.uniform(0.1, 0.4)
        final_score = min(1.0, stress_score + base_variation)
        
        return final_score
    
    def get_stress_level_text(self, score):
        """Convert stress score to readable text"""
        if score < STRESS_THRESHOLD_LOW:
            return "Low", "#4CAF50"  # Green
        elif score < STRESS_THRESHOLD_MODERATE:
            return "Moderate", "#FF9800"  # Orange
        elif score < STRESS_THRESHOLD_HIGH:
            return "High", "#FF5722"  # Deep Orange
        else:
            return "Very High", "#F44336"  # Red

class WebcamManager:
    """Manage webcam operations"""
    
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.stress_detector = SimpleStressDetector()
        self.current_frame = None
        self.current_stress_data = {
            'score': 0.0,
            'level': 'Low',
            'color': '#4CAF50',
            'timestamp': datetime.now().isoformat(),
            'face_detected': False
        }
        
    def start_camera(self):
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(WEBCAM_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, FPS)
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
                
            self.is_active = True
            logger.info("Camera started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_active = False
        if self.camera:
            self.camera.release()
            self.camera = None
        logger.info("Camera stopped")
    
    def process_frame(self):
        """Process single frame for stress detection"""
        if not self.is_active or not self.camera:
            return False
        
        ret, frame = self.camera.read()
        if not ret:
            return False
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        
        # Detect faces
        faces = self.stress_detector.detect_faces(frame)
        
        if len(faces) > 0:
            # Use first detected face
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            
            # Calculate stress level
            stress_score = self.stress_detector.calculate_basic_stress(face_roi)
            level_text, level_color = self.stress_detector.get_stress_level_text(stress_score)
            
            self.current_stress_data = {
                'score': round(stress_score, 2),
                'level': level_text,
                'color': level_color,
                'timestamp': datetime.now().isoformat(),
                'face_detected': True,
                'face_coords': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            }
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add stress level text
            cv2.putText(frame, f"Stress: {level_text} ({stress_score:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            self.current_stress_data['face_detected'] = False
            self.current_stress_data['timestamp'] = datetime.now().isoformat()
        
        return True
    
    def get_frame_jpeg(self):
        """Get current frame as JPEG bytes"""
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            return buffer.tobytes()
        return None

# Global webcam manager
webcam_manager = WebcamManager()

@app.route('/')
def index():
    """Main page"""
    return render_template('index_mvp.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start webcam monitoring"""
    try:
        success = webcam_manager.start_camera()
        if success:
            # Start processing thread
            thread = threading.Thread(target=continuous_processing)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'status': 'success',
                'message': 'Monitoring started successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to start camera'
            }), 500
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop webcam monitoring"""
    try:
        webcam_manager.stop_camera()
        return jsonify({
            'status': 'success',
            'message': 'Monitoring stopped'
        })
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/get_stress_data')
def get_stress_data():
    """Get current stress data"""
    return jsonify(webcam_manager.current_stress_data)

@app.route('/get_recommendations')
def get_recommendations():
    """Get wellness recommendations based on current stress level"""
    stress_level = webcam_manager.current_stress_data['level'].lower().replace(' ', '_')
    recommendations = RECOMMENDATIONS.get(stress_level, RECOMMENDATIONS['low_stress'])
    
    return jsonify({
        'stress_level': stress_level,
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while webcam_manager.is_active:
            frame_bytes = webcam_manager.get_frame_jpeg()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1/FPS)  # Control frame rate
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_session', methods=['POST'])
def save_session():
    """Save current session data"""
    try:
        data = request.get_json()
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': data.get('duration', 0),
            'avg_stress': data.get('avg_stress', 0),
            'max_stress': data.get('max_stress', 0),
            'notes': data.get('notes', '')
        }
        
        # Save to file
        sessions = []
        if SESSION_DATA_FILE.exists():
            with open(SESSION_DATA_FILE, 'r') as f:
                sessions = json.load(f)
        
        sessions.append(session_data)
        
        with open(SESSION_DATA_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        return jsonify({'status': 'success', 'message': 'Session saved'})
    
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_sessions')
def get_sessions():
    """Get saved sessions"""
    try:
        if not SESSION_DATA_FILE.exists():
            return jsonify({'sessions': []})
        
        with open(SESSION_DATA_FILE, 'r') as f:
            sessions = json.load(f)
        
        return jsonify({'sessions': sessions[-10:]})  # Return last 10 sessions
    
    except Exception as e:
        logger.error(f"Error loading sessions: {e}")
        return jsonify({'sessions': []})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'timestamp': datetime.now().isoformat(),
        'camera_active': webcam_manager.is_active
    })

def continuous_processing():
    """Continuous frame processing in background thread"""
    while webcam_manager.is_active:
        try:
            webcam_manager.process_frame()
            time.sleep(1/FPS)  # Control processing rate
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            time.sleep(1)  # Wait before retrying

if __name__ == '__main__':
    logger.info(f"Starting {APP_NAME} MVP v{VERSION}")
    app.run(debug=DEBUG, host=FLASK_HOST, port=FLASK_PORT)
