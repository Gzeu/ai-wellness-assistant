"""
Face Detection and Processing Utilities
Handles face detection, landmark extraction, and preprocessing
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional

class FaceProcessor:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face detection using OpenCV as fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_faces(self, frame):
        """Detect faces in frame and extract landmarks"""
        if frame is None:
            return []
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert normalized coordinates to pixel coordinates
                h, w = frame.shape[:2]
                landmarks = []
                
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append([x, y])
                
                faces.append({
                    'landmarks': landmarks,
                    'bounding_box': self._get_face_bounding_box(landmarks),
                    'confidence': 0.9  # MediaPipe confidence is generally high
                })
        
        return faces
    
    def _get_face_bounding_box(self, landmarks):
        """Calculate bounding box from landmarks"""
        landmarks = np.array(landmarks)
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        return {
            'x': int(x_min),
            'y': int(y_min), 
            'width': int(x_max - x_min),
            'height': int(y_max - y_min)
        }
    
    def preprocess_face(self, face_region):
        """Preprocess face region for analysis"""
        if face_region is None:
            return None
            
        # Resize to standard size
        processed = cv2.resize(face_region, (224, 224))
        
        # Normalize pixel values
        processed = processed.astype(np.float32) / 255.0
        
        # Histogram equalization for better contrast
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
        processed = cv2.equalizeHist((processed * 255).astype(np.uint8))
        processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0)):
        """Draw facial landmarks on frame for visualization"""
        for point in landmarks:
            cv2.circle(frame, tuple(point), 1, color, -1)
        return frame