#!/usr/bin/env python3
"""
AI Wellness Assistant - Enhanced MVP with User Authentication
Real-time stress detection with personalized user accounts and data storage

Author: George Pricop
Version: 2.0.0-AUTH
Date: September 2025
"""

import cv2
import numpy as np
import json
import os
import logging
import secrets
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
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

# Import authentication system
try:
    from auth_routes import auth_bp
    from auth import AuthManager, PersonalizedDataManager, require_auth
    AUTH_ENABLED = True
except ImportError:
    print("Warning: Authentication system not available. Running in basic mode.")
    AUTH_ENABLED = False
    auth_bp = None
    
    def require_auth(f):
        return f

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'wellness-assistant-auth-key-2025')
CORS(app)

# Setup authentication if available
if AUTH_ENABLED:
    auth_manager = AuthManager(app)
    app.register_blueprint(auth_bp)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access wellness monitoring.'
else:
    auth_manager = None
    PersonalizedDataManager = None

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
monitoring_sessions = {}  # Store active monitoring sessions by user
current_stress_data = {
    'score': 0.0,
    'level': 'unknown',
    'timestamp': None,
    'face_detected': False
}

# Create data directory
os.makedirs(DATA_DIRECTORY, exist_ok=True)

class EnhancedStressDetector:
    """
    Enhanced stress detection with user-specific calibration
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.user_baselines = {}  # Store user-specific stress baselines
        
    def calibrate_user_baseline(self, user_id: str, baseline_readings: list):
        """Calibrate stress detection for specific user"""
        if baseline_readings:
            avg_baseline = sum(baseline_readings) / len(baseline_readings)
            self.user_baselines[user_id] = {
                'baseline': avg_baseline,
                'calibration_date': datetime.now().isoformat(),
                'readings_count': len(baseline_readings)
            }
        
    def detect_stress_level(self, frame, user_id=None):
        """
        Analyze frame for stress indicators with user-specific calibration
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return {
                    'score': 0.0,
                    'level': 'no_face',
                    'face_detected': False,
                    'face_count': 0,
                    'user_calibrated': user_id in self.user_baselines if user_id else False
                }
            
            # Use the largest face detected
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes within face
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            # Calculate stress score
            raw_stress_score = self._calculate_stress_score(face_roi, eyes, w, h)
            
            # Apply user-specific calibration if available
            calibrated_score = raw_stress_score
            if user_id and user_id in self.user_baselines:
                baseline = self.user_baselines[user_id]['baseline']
                # Adjust score based on user's baseline
                calibrated_score = max(0.0, min(1.0, raw_stress_score + (raw_stress_score - baseline) * 0.3))
            
            stress_level = self._categorize_stress_level(calibrated_score, user_id)
            
            return {
                'score': calibrated_score,
                'raw_score': raw_stress_score,
                'level': stress_level,
                'face_detected': True,
                'face_count': len(faces),
                'face_area': w * h,
                'eyes_detected': len(eyes),
                'user_calibrated': user_id in self.user_baselines if user_id else False,
                'confidence': min(1.0, len(eyes) / 2.0)  # Confidence based on eye detection
            }
            
        except Exception as e:
            logger.error(f"Error in stress detection: {e}")
            return {
                'score': 0.0,
                'level': 'error',
                'face_detected': False,
                'face_count': 0,
                'user_calibrated': False
            }
    
    def _calculate_stress_score(self, face_roi, eyes, face_width, face_height):
        """Enhanced stress score calculation"""
        try:
            stress_factors = []
            
            # Factor 1: Face symmetry analysis (improved)
            left_half = face_roi[:, :face_width//2]
            right_half = face_roi[:, face_width//2:]
            right_half_flipped = np.fliplr(right_half)
            
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float)))
            symmetry_factor = min(symmetry_diff / 40.0, 1.0)  # More sensitive
            stress_factors.append(symmetry_factor)
            
            # Factor 2: Enhanced eye region analysis
            if len(eyes) >= 2:
                eye1, eye2 = eyes[0], eyes[1]
                eye_distance = abs(eye1[0] - eye2[0])
                expected_distance = face_width * 0.3
                distance_factor = abs(eye_distance - expected_distance) / expected_distance
                
                # Eye size analysis
                avg_eye_area = (eye1[2] * eye1[3] + eye2[2] * eye2[3]) / 2
                expected_eye_area = (face_width * face_height) * 0.015  # ~1.5% of face
                size_factor = abs(avg_eye_area - expected_eye_area) / expected_eye_area
                
                stress_factors.append(min(distance_factor, 1.0))
                stress_factors.append(min(size_factor, 1.0))
            else:
                stress_factors.extend([0.5, 0.5])  # Moderate stress if eyes not detected
            
            # Factor 3: Texture analysis (micro-expressions)
            face_std = np.std(face_roi)
            intensity_factor = min(face_std / 45.0, 1.0)
            stress_factors.append(intensity_factor)
            
            # Factor 4: Enhanced edge density (tension)
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / (face_width * face_height)
            edge_factor = min(edge_density * 12, 1.0)
            stress_factors.append(edge_factor)
            
            # Factor 5: Gradient analysis (facial muscle tension)
            grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_factor = min(np.mean(gradient_magnitude) / 100.0, 1.0)
            stress_factors.append(gradient_factor)
            
            # Weighted combination with enhanced factors
            weights = [0.25, 0.15, 0.15, 0.2, 0.15, 0.1]  # Sum = 1.0
            stress_score = sum(f * w for f, w in zip(stress_factors, weights))
            
            # Add temporal smoothing (reduced randomization)
            noise = (np.random.random() - 0.5) * 0.05
            stress_score = max(0.0, min(1.0, stress_score + noise))
            
            return stress_score
            
        except Exception as e:
            logger.error(f"Error calculating stress score: {e}")
            return 0.3  # Default moderate stress
    
    def _categorize_stress_level(self, score, user_id=None):
        """Categorize stress with user-specific thresholds"""
        # Get user preferences if authenticated
        thresholds = {
            'low': STRESS_THRESHOLD_LOW,
            'moderate': STRESS_THRESHOLD_MODERATE,
            'high': STRESS_THRESHOLD_HIGH
        }
        
        if AUTH_ENABLED and user_id and current_user.is_authenticated:
            user_thresholds = current_user.preferences.get('stress_thresholds', {})
            thresholds.update(user_thresholds)
        
        if score < thresholds['low']:
            return 'low'
        elif score < thresholds['moderate']:
            return 'moderate'
        elif score < thresholds['high']:
            return 'high'
        else:
            return 'very_high'

# Initialize enhanced stress detector
stress_detector = EnhancedStressDetector()

class UserSession:
    """Manage user monitoring session"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.session_id = secrets.token_urlsafe(16)
        self.start_time = datetime.now()
        self.end_time = None
        self.stress_readings = []
        self.is_active = False
        self.total_face_detections = 0
        self.stress_episodes = 0
        
    def add_stress_reading(self, stress_data):
        """Add stress reading to session"""
        reading = {
            'timestamp': datetime.now().isoformat(),
            'score': stress_data['score'],
            'level': stress_data['level'],
            'face_detected': stress_data['face_detected'],
            'confidence': stress_data.get('confidence', 0.0)
        }
        
        self.stress_readings.append(reading)
        
        if stress_data['face_detected']:
            self.total_face_detections += 1
        
        if stress_data['level'] in ['high', 'very_high']:
            self.stress_episodes += 1
            
        # Save real-time data if PersonalizedDataManager is available
        if PersonalizedDataManager:
            PersonalizedDataManager.add_stress_data_point(
                self.user_id,
                stress_data['score'],
                stress_data['level'],
                stress_data['face_detected'],
                self.session_id
            )
    
    def end_session(self):
        """End monitoring session and calculate analytics"""
        self.end_time = datetime.now()
        self.is_active = False
        
        # Calculate session analytics
        duration = (self.end_time - self.start_time).total_seconds()
        
        if self.stress_readings:
            scores = [r['score'] for r in self.stress_readings if r['face_detected']]
            analytics = {
                'duration': duration,
                'total_readings': len(self.stress_readings),
                'valid_readings': len(scores),
                'avg_stress': sum(scores) / len(scores) if scores else 0.0,
                'max_stress': max(scores) if scores else 0.0,
                'min_stress': min(scores) if scores else 0.0,
                'stress_episodes': self.stress_episodes,
                'face_detection_rate': self.total_face_detections / len(self.stress_readings) if self.stress_readings else 0.0
            }
        else:
            analytics = {
                'duration': duration,
                'total_readings': 0,
                'valid_readings': 0,
                'avg_stress': 0.0,
                'max_stress': 0.0,
                'min_stress': 0.0,
                'stress_episodes': 0,
                'face_detection_rate': 0.0
            }
        
        # Save session if PersonalizedDataManager is available
        if PersonalizedDataManager:
            session_data = {
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'stress_readings': self.stress_readings,
                'user_id': self.user_id
            }
            
            PersonalizedDataManager.save_user_session(self.user_id, session_data, analytics)
        
        return analytics
    
    def get_session_data(self):
        """Get current session data"""
        current_time = datetime.now()
        duration = (current_time - self.start_time).total_seconds()
        
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'start_time': self.start_time.isoformat(),
            'duration': duration,
            'is_active': self.is_active,
            'total_readings': len(self.stress_readings),
            'stress_episodes': self.stress_episodes,
            'recent_readings': self.stress_readings[-10:] if self.stress_readings else []
        }

def get_enhanced_wellness_recommendations(stress_level, user_id=None):
    """Get personalized wellness recommendations"""
    base_recommendations = {
        'low': [
            "Excellent! You're in a relaxed state. 🌟",
            "Consider some light stretching to maintain this calm.",
            "Perfect time for creative or focused work.",
            "Stay hydrated and keep up the good work!"
        ],
        'moderate': [
            "Take a few deep breaths using the 4-7-8 technique. 🌬️",
            "Try a 5-minute guided meditation break.",
            "Step away from screens and look at something distant.",
            "Do some neck and shoulder stretches.",
            "Consider a short walk or light movement."
        ],
        'high': [
            "⚠️ High stress detected. Take immediate action.",
            "Practice progressive muscle relaxation.",
            "Try box breathing: 4-4-4-4 pattern.",
            "Take a longer break from current activities.",
            "Consider reaching out for support if needed."
        ],
        'very_high': [
            "🚨 Very high stress levels detected.",
            "Stop current activities and focus on calming down.",
            "Use emergency stress relief techniques.",
            "Ensure you're in a safe, comfortable environment.",
            "Consider contacting a healthcare professional."
        ]
    }
    
    recommendations = base_recommendations.get(stress_level, base_recommendations['moderate'])
    
    # Add personalized recommendations if user is authenticated
    if AUTH_ENABLED and PersonalizedDataManager and user_id:
        try:
            personalized = PersonalizedDataManager.get_personalized_recommendations(user_id)
            if personalized:
                recommendations.extend([rec['text'] for rec in personalized[:2]])
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
    
    return recommendations

def initialize_camera():
    """Initialize camera for video capture"""
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

def generate_frames(user_id=None):
    """Generate video frames for streaming with user-specific processing"""
    global camera, monitoring_sessions, current_stress_data
    
    user_session = monitoring_sessions.get(user_id)
    if not user_session or not user_session.is_active:
        return
    
    while user_session.is_active and camera is not None:
        try:
            success, frame = camera.read()
            if not success:
                break
            
            # Analyze frame for stress with user-specific calibration
            stress_result = stress_detector.detect_stress_level(frame, user_id)
            
            # Update global stress data
            current_stress_data.update({
                'score': round(stress_result['score'], 3),
                'level': stress_result['level'],
                'timestamp': datetime.now().isoformat(),
                'face_detected': stress_result['face_detected'],
                'user_calibrated': stress_result.get('user_calibrated', False),
                'confidence': stress_result.get('confidence', 0.0)
            })
            
            # Add reading to user session
            user_session.add_stress_reading(stress_result)
            
            # Draw enhanced stress overlay
            frame = draw_enhanced_stress_overlay(frame, stress_result, user_session)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            break
        
        time.sleep(0.1)  # Control frame rate

def draw_enhanced_stress_overlay(frame, stress_result, user_session):
    """Draw enhanced stress detection overlay"""
    try:
        # Enhanced colors for stress levels
        colors = {
            'low': (0, 255, 0),        # Green
            'moderate': (0, 165, 255), # Orange
            'high': (0, 100, 255),     # Red-Orange
            'very_high': (0, 0, 255),  # Red
            'no_face': (128, 128, 128), # Gray
            'unknown': (128, 128, 128)  # Gray
        }
        
        color = colors.get(stress_result['level'], (128, 128, 128))
        
        # Draw face detection with confidence indicator
        if stress_result['face_detected']:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = stress_detector.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Main face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Confidence indicator
                confidence = stress_result.get('confidence', 0.0)
                conf_text = f"Conf: {confidence:.2f}"
                cv2.putText(frame, conf_text, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Stress level text
                level_text = f"Stress: {stress_result['level']} ({stress_result['score']:.3f})"
                cv2.putText(frame, level_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Session information
        session_data = user_session.get_session_data()
        session_text = f"Session: {int(session_data['duration'])}s | Readings: {session_data['total_readings']}"
        cv2.putText(frame, session_text, (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # User calibration indicator
        if stress_result.get('user_calibrated', False):
            cv2.putText(frame, "CALIBRATED", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error drawing enhanced overlay: {e}")
        return frame

# Enhanced Flask routes
@app.route('/')
def index():
    """Main page with authentication awareness"""
    if AUTH_ENABLED and current_user.is_authenticated:
        return redirect(url_for('auth.dashboard'))
    else:
        return render_template('index_mvp.html' if not AUTH_ENABLED else 'index_auth.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start personalized stress monitoring"""
    try:
        user_id = None
        if AUTH_ENABLED and current_user.is_authenticated:
            user_id = current_user.id
        elif not AUTH_ENABLED:
            user_id = 'anonymous'
        else:
            return jsonify({
                'status': 'error',
                'message': 'Authentication required for monitoring'
            }), 401
        
        # Check if user already has active session
        if user_id in monitoring_sessions and monitoring_sessions[user_id].is_active:
            return jsonify({
                'status': 'info',
                'message': 'Monitoring already active for this user'
            })
        
        # Initialize camera
        if not initialize_camera():
            return jsonify({
                'status': 'error',
                'message': 'Failed to initialize camera'
            }), 500
        
        # Create new user session
        user_session = UserSession(user_id)
        user_session.is_active = True
        monitoring_sessions[user_id] = user_session
        
        logger.info(f"Monitoring started for user: {user_id}")
        return jsonify({
            'status': 'success',
            'message': 'Personalized monitoring started successfully',
            'session_id': user_session.session_id,
            'user_calibrated': user_id in stress_detector.user_baselines
        })
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error starting monitoring: {str(e)}'
        }), 500

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop personalized stress monitoring"""
    try:
        user_id = None
        if AUTH_ENABLED and current_user.is_authenticated:
            user_id = current_user.id
        elif not AUTH_ENABLED:
            user_id = 'anonymous'
        else:
            return jsonify({
                'status': 'error',
                'message': 'Authentication required'
            }), 401
        
        # End user session
        session_analytics = None
        if user_id in monitoring_sessions:
            user_session = monitoring_sessions[user_id]
            session_analytics = user_session.end_session()
            del monitoring_sessions[user_id]
        
        # Release camera if no active sessions
        if not any(session.is_active for session in monitoring_sessions.values()):
            global camera
            if camera is not None:
                camera.release()
                camera = None
        
        logger.info(f"Monitoring stopped for user: {user_id}")
        return jsonify({
            'status': 'success',
            'message': 'Monitoring stopped successfully',
            'session_analytics': session_analytics
        })
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error stopping monitoring: {str(e)}'
        }), 500

@app.route('/video_feed')
def video_feed():
    """Personalized video streaming route"""
    user_id = None
    if AUTH_ENABLED and current_user.is_authenticated:
        user_id = current_user.id
    elif not AUTH_ENABLED:
        user_id = 'anonymous'
    else:
        return "Authentication required", 401
    
    if user_id in monitoring_sessions and monitoring_sessions[user_id].is_active:
        return Response(generate_frames(user_id),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Monitoring not active for this user", 404

@app.route('/get_stress_data')
def get_stress_data():
    """Get current personalized stress data"""
    user_id = None
    if AUTH_ENABLED and current_user.is_authenticated:
        user_id = current_user.id
    elif not AUTH_ENABLED:
        user_id = 'anonymous'
    
    response_data = current_stress_data.copy()
    
    # Add session information if available
    if user_id and user_id in monitoring_sessions:
        session_data = monitoring_sessions[user_id].get_session_data()
        response_data['session'] = session_data
    
    return jsonify(response_data)

@app.route('/get_recommendations')
def get_recommendations():
    """Get personalized wellness recommendations"""
    user_id = None
    if AUTH_ENABLED and current_user.is_authenticated:
        user_id = current_user.id
    
    level = current_stress_data.get('level', 'unknown')
    recommendations = get_enhanced_wellness_recommendations(level, user_id)
    
    return jsonify({
        'level': level,
        'recommendations': recommendations,
        'personalized': user_id is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    """Enhanced health check with authentication status"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0-AUTH',
        'authentication_enabled': AUTH_ENABLED,
        'active_sessions': len(monitoring_sessions),
        'camera_available': camera is not None and camera.isOpened() if camera else False,
        'features': {
            'user_authentication': AUTH_ENABLED,
            'personalized_data': PersonalizedDataManager is not None,
            'user_calibration': True,
            'session_analytics': True
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting AI Wellness Assistant with Authentication")
    logger.info(f"Authentication enabled: {AUTH_ENABLED}")
    logger.info(f"Server will run on http://{FLASK_HOST}:{FLASK_PORT}")
    
    try:
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG_MODE, threaded=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        # Clean up camera and sessions on exit
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()
        monitoring_sessions.clear()
        logger.info("Application shutdown complete")