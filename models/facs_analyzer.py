"""
Facial Action Coding System (FACS) Analyzer
Detects and quantifies facial action units for emotion and stress analysis
Based on Ekman & Friesen FACS methodology
"""

import numpy as np
import math
from datetime import datetime

class FACSAnalyzer:
    def __init__(self):
        # FACS Action Unit definitions with landmark relationships
        self.action_unit_definitions = {
            'AU1': {  # Inner Brow Raiser
                'name': 'Inner Brow Raiser',
                'landmarks': [17, 18, 19, 20],
                'description': 'Raises inner portion of eyebrows',
                'muscles': ['frontalis_medial']
            },
            'AU2': {  # Outer Brow Raiser
                'name': 'Outer Brow Raiser', 
                'landmarks': [21, 22, 23, 24, 25, 26],
                'description': 'Raises outer portion of eyebrows',
                'muscles': ['frontalis_lateral']
            },
            'AU4': {  # Brow Lowerer
                'name': 'Brow Lowerer',
                'landmarks': [17, 18, 19, 20, 21, 22],
                'description': 'Lowers and draws eyebrows together',
                'muscles': ['corrugator', 'procerus']
            },
            'AU7': {  # Lid Tightener
                'name': 'Lid Tightener',
                'landmarks': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                'description': 'Tightens eyelids',
                'muscles': ['orbicularis_oculi']
            },
            'AU15': {  # Lip Corner Depressor
                'name': 'Lip Corner Depressor',
                'landmarks': [48, 54],
                'description': 'Pulls lip corners downward',
                'muscles': ['depressor_anguli_oris']
            },
            'AU23': {  # Lip Tightener
                'name': 'Lip Tightener',
                'landmarks': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                'description': 'Tightens and narrows lips',
                'muscles': ['orbicularis_oris']
            }
        }
        
        self.reference_measurements = None
        self.calibration_frames = []
        
    def extract_action_units(self, face_landmarks):
        """Extract and quantify facial action units from facial landmarks"""
        if face_landmarks is None or len(face_landmarks) != 68:
            return {}
            
        # Convert landmarks to numpy array for easier processing
        landmarks = np.array(face_landmarks)
        
        # Normalize landmarks relative to face size and position
        normalized_landmarks = self._normalize_landmarks(landmarks)
        
        # Extract action units
        action_units = {}
        
        for au_code, au_info in self.action_unit_definitions.items():
            intensity = self._calculate_au_intensity(au_code, normalized_landmarks)
            action_units[au_code] = intensity
            
        return action_units
    
    def _normalize_landmarks(self, landmarks):
        """Normalize landmarks to account for face size and position"""
        # Calculate face center
        face_center = np.mean(landmarks, axis=0)
        
        # Calculate face width (distance between jaw points)
        face_width = np.linalg.norm(landmarks[16] - landmarks[0])
        
        # Normalize landmarks
        normalized = (landmarks - face_center) / face_width
        
        return normalized
    
    def _calculate_au_intensity(self, au_code, landmarks):
        """Calculate the intensity of a specific action unit"""
        au_info = self.action_unit_definitions[au_code]
        
        # Generic calculation based on landmark displacement
        relevant_landmarks = [landmarks[i] for i in au_info['landmarks']]
        landmark_array = np.array(relevant_landmarks)
        variance = np.var(landmark_array)
        
        # Scale to 0-1 range
        intensity = min(1.0, variance * 100)
        
        return max(0.0, min(1.0, intensity))