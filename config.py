"""Configuration file for AI Wellness Assistant MVP"""
import os
from pathlib import Path

# Application settings
APP_NAME = "AI Wellness Assistant"
VERSION = "0.1.0-MVP"
DEBUG = True

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Webcam settings
WEBCAM_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Stress detection settings
STRESS_DETECTION_ENABLED = True
STRESS_THRESHOLD_LOW = 0.3
STRESS_THRESHOLD_MODERATE = 0.6
STRESS_THRESHOLD_HIGH = 0.8

# Face detection settings
FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_SIZE = (50, 50)
MAX_FACES = 1  # MVP supports single face

# Data storage settings
DATA_DIR = Path("data")
SESSION_DATA_FILE = DATA_DIR / "sessions.json"
JOURNAL_DATA_FILE = DATA_DIR / "journal_entries.json"
USER_PREFERENCES_FILE = DATA_DIR / "preferences.json"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# API settings
API_RATE_LIMIT = 100  # requests per minute
API_TIMEOUT = 30  # seconds

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/app.log"

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Wellness recommendations
RECOMMENDATIONS = {
    "low_stress": [
        "Great job maintaining a calm state!",
        "Consider some light stretching to stay relaxed.",
        "Take a moment to appreciate your current peaceful state."
    ],
    "moderate_stress": [
        "Try taking 5 deep breaths: inhale for 4 counts, hold for 4, exhale for 6.",
        "Consider a short walk or some light physical activity.",
        "Take a 2-minute break from your current task.",
        "Practice progressive muscle relaxation."
    ],
    "high_stress": [
        "Stop what you're doing and take 10 deep breaths.",
        "Try the 4-7-8 breathing technique: inhale 4, hold 7, exhale 8.",
        "Consider stepping away from stressful tasks for 10-15 minutes.",
        "Practice mindfulness: focus on your immediate surroundings.",
        "Drink some water and do gentle neck rolls."
    ],
    "very_high_stress": [
        "IMMEDIATE ACTION: Take a break from all activities.",
        "Practice emergency breathing: 6 slow, deep breaths.",
        "Find a quiet space and sit comfortably.",
        "Consider speaking with someone you trust.",
        "If this persists, consider professional support."
    ]
}
