"""
AI Wellness Assistant - Real-time Stress Detection from Facial Micro-expressions
Author: Pricop George
Description: Advanced AI system for detecting stress levels using FACS and providing wellness guidance
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from datetime import datetime
import json
import threading
import time
from models.stress_detector import StressDetector
from models.facs_analyzer import FACSAnalyzer
from models.wellness_coach import WellnessCoach
from utils.face_utils import FaceProcessor
from utils.emotion_utils import EmotionAnalyzer
from utils.journal_utils import JournalManager

app = Flask(__name__)

# Initialize core components
stress_detector = StressDetector()
facs_analyzer = FACSAnalyzer()
wellness_coach = WellnessCoach()
face_processor = FaceProcessor()
emotion_analyzer = EmotionAnalyzer()
journal_manager = JournalManager()

# Global variables for real-time processing
current_stress_score = 0.0
last_analysis_time = datetime.now()
stress_trend = []

class WebcamStressAnalyzer:
    def __init__(self):
        self.camera = None
        self.is_analyzing = False
        self.frame_count = 0
        
    def start_analysis(self):
        """Start real-time stress analysis from webcam"""
        self.camera = cv2.VideoCapture(0)
        self.is_analyzing = True
        
        while self.is_analyzing:
            success, frame = self.camera.read()
            if not success:
                break
            
            # Process frame for stress detection
            self.analyze_frame(frame)
            
    def analyze_frame(self, frame):
        """Analyze single frame for stress indicators"""
        global current_stress_score, stress_trend
        
        # Face detection and preprocessing
        faces = face_processor.detect_faces(frame)
        
        for face in faces:
            # Extract facial action units using FACS
            action_units = facs_analyzer.extract_action_units(face)
            
            # Analyze micro-expressions for stress indicators
            stress_indicators = stress_detector.analyze_stress_signals(action_units)
            
            # Calculate normalized stress score (0-1)
            stress_score = stress_detector.calculate_stress_score(stress_indicators)
            
            # Update global stress tracking
            current_stress_score = stress_score
            stress_trend.append({
                'timestamp': datetime.now().isoformat(),
                'score': stress_score,
                'indicators': stress_indicators
            })
            
            # Keep only last 100 measurements
            if len(stress_trend) > 100:
                stress_trend.pop(0)
                
            # Trigger alerts for high stress
            if stress_score > 0.7:
                self.handle_high_stress_alert(stress_score, stress_indicators)
    
    def handle_high_stress_alert(self, score, indicators):
        """Handle high stress detection"""
        recommendations = wellness_coach.get_stress_relief_techniques(score, indicators)
        # Could send notifications, log events, etc.
        
    def stop_analysis(self):
        """Stop the analysis"""
        self.is_analyzing = False
        if self.camera:
            self.camera.release()

# Initialize webcam analyzer
webcam_analyzer = WebcamStressAnalyzer()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start real-time stress monitoring"""
    thread = threading.Thread(target=webcam_analyzer.start_analysis)
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'monitoring_started'})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop stress monitoring"""
    webcam_analyzer.stop_analysis()
    return jsonify({'status': 'monitoring_stopped'})

@app.route('/get_stress_data')
def get_stress_data():
    """Get current stress data"""
    return jsonify({
        'current_score': current_stress_score,
        'trend': stress_trend[-20:],  # Last 20 measurements
        'timestamp': datetime.now().isoformat(),
        'status': 'active' if webcam_analyzer.is_analyzing else 'inactive'
    })

@app.route('/get_wellness_recommendations')
def get_wellness_recommendations():
    """Get personalized wellness recommendations"""
    recommendations = wellness_coach.get_recommendations(current_stress_score, stress_trend)
    return jsonify(recommendations)

@app.route('/journal')
def journal():
    """Emotional journal interface"""
    entries = journal_manager.get_recent_entries()
    return render_template('journal.html', entries=entries)

@app.route('/add_journal_entry', methods=['POST'])
def add_journal_entry():
    """Add new journal entry"""
    data = request.get_json()
    entry = journal_manager.add_entry(
        mood=data.get('mood'),
        stress_level=data.get('stress_level'),
        notes=data.get('notes'),
        highlights=data.get('highlights')
    )
    return jsonify({'status': 'entry_added', 'entry_id': entry['id']})

@app.route('/dashboard')
def dashboard():
    """Wellness dashboard with insights"""
    insights = journal_manager.generate_insights()
    return render_template('dashboard.html', insights=insights)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)