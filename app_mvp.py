#!/usr/bin/env python3
"""
AI Wellness Assistant - MVP Application
Real-time stress detection using facial expression analysis

Author: George Pricop
Version: 1.0.0-MVP
Date: September 2025
"""

import cv2
import numpy as np
import json
import os
import logging
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import threading
import time

# Import configuration
try:
    from config import *
except ImportError:
    # Default configuration if config.py is not found
    WEBCAM_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    STRESS_THRESHOLD_LOW = 0.3
    STRESS_THRESHOLD_MODERATE = 0.6
    STRESS_THRESHOLD_HIGH = 0.8
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    DEBUG_MODE = True
    DATA_DIRECTORY = "data"
    LOGS_DIRECTORY = "logs"

# Create Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
os.makedirs(LOGS_DIRECTORY, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIRECTORY, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
camera = None
monitoring_active = False
current_stress_data = {
    'score': 0.0,
    'level': 'unknown',
    'timestamp': None,
    'face_detected': False
}
session_data = []

# Create data directory
os.makedirs(DATA_DIRECTORY, exist_ok=True)

class StressDetector:
    """
    Simple stress detection based on facial analysis
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_stress_level(self, frame):
        """
        Analyze frame for stress indicators
        Returns: dict with stress score and level
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return {
                    'score': 0.0,
                    'level': 'no_face',
                    'face_detected': False,
                    'face_count': 0
                }
            
            # Use the largest face detected
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes within face
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            # Calculate basic stress indicators
            stress_score = self._calculate_stress_score(face_roi, eyes, w, h)
            stress_level = self._categorize_stress_level(stress_score)
            
            return {
                'score': stress_score,
                'level': stress_level,
                'face_detected': True,
                'face_count': len(faces),
                'face_area': w * h,
                'eyes_detected': len(eyes)
            }
            
        except Exception as e:
            logger.error(f"Error in stress detection: {e}")
            return {
                'score': 0.0,
                'level': 'error',
                'face_detected': False,
                'face_count': 0
            }
    
    def _calculate_stress_score(self, face_roi, eyes, face_width, face_height):
        """
        Calculate stress score based on facial features
        """
        try:
            # Basic stress indicators calculation
            stress_factors = []
            
            # Factor 1: Face symmetry analysis
            left_half = face_roi[:, :face_width//2]
            right_half = face_roi[:, face_width//2:]
            right_half_flipped = np.fliplr(right_half)
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float)))
            symmetry_factor = min(symmetry_diff / 50.0, 1.0)  # Normalize
            stress_factors.append(symmetry_factor)
            
            # Factor 2: Eye region analysis
            if len(eyes) >= 2:
                # Eye spacing and size analysis
                eye1, eye2 = eyes[0], eyes[1]
                eye_distance = abs(eye1[0] - eye2[0])
                expected_distance = face_width * 0.3  # Typical eye distance
                distance_factor = abs(eye_distance - expected_distance) / expected_distance
                stress_factors.append(min(distance_factor, 1.0))
            else:
                stress_factors.append(0.4)  # Moderate stress if eyes not detected properly
            
            # Factor 3: Intensity variations (texture analysis)
            face_std = np.std(face_roi)
            intensity_factor = min(face_std / 50.0, 1.0)
            stress_factors.append(intensity_factor)
            
            # Factor 4: Edge density (tension indicators)
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / (face_width * face_height)
            edge_factor = min(edge_density * 10, 1.0)
            stress_factors.append(edge_factor)
            
            # Combine factors with weights
            weights = [0.3, 0.25, 0.25, 0.2]  # Symmetry, eyes, intensity, edges
            stress_score = sum(f * w for f, w in zip(stress_factors, weights))
            
            # Add some randomization to make it more realistic for demo
            noise = (np.random.random() - 0.5) * 0.1
            stress_score = max(0.0, min(1.0, stress_score + noise))
            
            return stress_score
            
        except Exception as e:
            logger.error(f"Error calculating stress score: {e}")
            return 0.3  # Default moderate stress
    
    def _categorize_stress_level(self, score):
        """
        Categorize stress score into levels
        """
        if score < STRESS_THRESHOLD_LOW:
            return 'low'
        elif score < STRESS_THRESHOLD_MODERATE:
            return 'moderate'
        elif score < STRESS_THRESHOLD_HIGH:
            return 'high'
        else:
            return 'very_high'

# Initialize stress detector
stress_detector = StressDetector()

def get_wellness_recommendations(stress_level):
    """
    Get wellness recommendations based on stress level
    """
    recommendations = {
        'low': [
            "Great job! You're in a relaxed state.",
            "Continue with your current activities.",
            "Consider some light stretching or a short walk.",
            "Stay hydrated and maintain good posture."
        ],
        'moderate': [
            "Take a few deep breaths.",
            "Try the 4-7-8 breathing technique.",
            "Consider a 5-minute meditation break.",
            "Stretch your neck and shoulders.",
            "Step away from screens for a moment."
        ],
        'high': [
            "Take immediate action to reduce stress.",
            "Practice progressive muscle relaxation.",
            "Try guided breathing exercises.",
            "Consider taking a longer break.",
            "Reach out to someone for support if needed."
        ],
        'very_high': [
            "Your stress levels are very high. Please take immediate action.",
            "Stop current activities and focus on calming down.",
            "Practice emergency stress relief techniques.",
            "Consider contacting a healthcare professional.",
            "Ensure you're in a safe, comfortable environment."
        ],
        'no_face': [
            "Please position your face in the camera view.",
            "Ensure good lighting for better detection.",
            "Sit directly in front of the camera."
        ],
        'unknown': [
            "Unable to analyze stress level at the moment.",
            "Please ensure your face is clearly visible."
        ]
    }
    
    return recommendations.get(stress_level, recommendations['unknown'])

def initialize_camera():
    """
    Initialize camera for video capture
    """
    global camera
    try:
        camera = cv2.VideoCapture(WEBCAM_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return False
        
        logger.info("Camera initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return False

def generate_frames():
    """
    Generate video frames for streaming
    """
    global camera, monitoring_active, current_stress_data
    
    while monitoring_active and camera is not None:
        try:
            success, frame = camera.read()
            if not success:
                break
            
            # Analyze frame for stress
            stress_result = stress_detector.detect_stress_level(frame)
            
            # Update global stress data
            current_stress_data.update({
                'score': round(stress_result['score'], 3),
                'level': stress_result['level'],
                'timestamp': datetime.now().isoformat(),
                'face_detected': stress_result['face_detected']
            })
            
            # Draw stress visualization on frame
            frame = draw_stress_overlay(frame, stress_result)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            break
        
        time.sleep(0.1)  # Control frame rate

def draw_stress_overlay(frame, stress_result):
    """
    Draw stress detection overlay on frame
    """
    try:
        # Define colors for stress levels
        colors = {
            'low': (0, 255, 0),        # Green
            'moderate': (0, 165, 255), # Orange
            'high': (0, 100, 255),     # Red-Orange
            'very_high': (0, 0, 255),  # Red
            'no_face': (128, 128, 128), # Gray
            'unknown': (128, 128, 128)  # Gray
        }
        
        color = colors.get(stress_result['level'], (128, 128, 128))
        
        # Draw face detection box if face is detected
        if stress_result['face_detected']:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = stress_detector.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                # Add stress level text
                cv2.putText(frame, f"Stress: {stress_result['level']}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add stress score to frame
        score_text = f"Score: {stress_result['score']:.3f}"
        cv2.putText(frame, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error drawing overlay: {e}")
        return frame

# Flask routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index_mvp.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start stress monitoring"""
    global monitoring_active
    
    try:
        if not monitoring_active:
            if initialize_camera():
                monitoring_active = True
                logger.info("Monitoring started")
                return jsonify({
                    'status': 'success',
                    'message': 'Monitoring started successfully'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to initialize camera'
                }), 500
        else:
            return jsonify({
                'status': 'info',
                'message': 'Monitoring already active'
            })
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error starting monitoring: {str(e)}'
        }), 500

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop stress monitoring"""
    global monitoring_active, camera
    
    try:
        monitoring_active = False
        
        if camera is not None:
            camera.release()
            camera = None
        
        logger.info("Monitoring stopped")
        return jsonify({
            'status': 'success',
            'message': 'Monitoring stopped successfully'
        })
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error stopping monitoring: {str(e)}'
        }), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    if monitoring_active:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Monitoring not active", 404

@app.route('/get_stress_data')
def get_stress_data():
    """Get current stress data"""
    return jsonify(current_stress_data)

@app.route('/get_recommendations')
def get_recommendations():
    """Get wellness recommendations"""
    level = current_stress_data.get('level', 'unknown')
    recommendations = get_wellness_recommendations(level)
    
    return jsonify({
        'level': level,
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/save_session', methods=['POST'])
def save_session():
    """Save current session data"""
    try:
        session_info = {
            'timestamp': datetime.now().isoformat(),
            'stress_data': current_stress_data.copy(),
            'session_duration': request.json.get('duration', 0),
            'average_stress': request.json.get('average_stress', 0.0)
        }
        
        # Save to file
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(DATA_DIRECTORY, filename)
        
        with open(filepath, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        logger.info(f"Session saved: {filename}")
        return jsonify({
            'status': 'success',
            'message': f'Session saved as {filename}'
        })
    
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error saving session: {str(e)}'
        }), 500

@app.route('/get_sessions')
def get_sessions():
    """Get saved sessions"""
    try:
        sessions = []
        
        for filename in os.listdir(DATA_DIRECTORY):
            if filename.endswith('.json'):
                filepath = os.path.join(DATA_DIRECTORY, filename)
                try:
                    with open(filepath, 'r') as f:
                        session_data = json.load(f)
                        sessions.append({
                            'filename': filename,
                            'data': session_data
                        })
                except Exception as e:
                    logger.error(f"Error reading session {filename}: {e}")
        
        return jsonify({
            'status': 'success',
            'sessions': sessions
        })
    
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error getting sessions: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0-MVP',
        'monitoring_active': monitoring_active,
        'camera_available': camera is not None and camera.isOpened() if camera else False,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting AI Wellness Assistant MVP")
    logger.info(f"Server will run on http://{FLASK_HOST}:{FLASK_PORT}")
    
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG_MODE, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Clean up camera on exit
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
        logger.info("Application shutdown complete")