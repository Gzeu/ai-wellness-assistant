"""
Advanced Stress Detection using Facial Action Coding System (FACS)
Implements real-time stress analysis from micro-expressions
"""

import numpy as np
from datetime import datetime, timedelta
import math

class StressDetector:
    def __init__(self):
        # FACS Action Units associated with stress indicators
        self.stress_action_units = {
            'AU1': {'name': 'Inner Brow Raiser', 'stress_weight': 0.7},
            'AU2': {'name': 'Outer Brow Raiser', 'stress_weight': 0.6},
            'AU4': {'name': 'Brow Lowerer', 'stress_weight': 0.8},
            'AU5': {'name': 'Upper Lid Raiser', 'stress_weight': 0.5},
            'AU7': {'name': 'Lid Tightener', 'stress_weight': 0.6},
            'AU9': {'name': 'Nose Wrinkler', 'stress_weight': 0.4},
            'AU10': {'name': 'Upper Lip Raiser', 'stress_weight': 0.3},
            'AU11': {'name': 'Nasolabial Deepener', 'stress_weight': 0.4},
            'AU14': {'name': 'Dimpler', 'stress_weight': 0.5},
            'AU15': {'name': 'Lip Corner Depressor', 'stress_weight': 0.7},
            'AU17': {'name': 'Chin Raiser', 'stress_weight': 0.4},
            'AU20': {'name': 'Lip Stretcher', 'stress_weight': 0.6},
            'AU22': {'name': 'Lip Funneler', 'stress_weight': 0.3},
            'AU23': {'name': 'Lip Tightener', 'stress_weight': 0.5},
            'AU24': {'name': 'Lip Pressor', 'stress_weight': 0.6},
        }
        
        # Physiological indicators of stress
        self.stress_patterns = {
            'tension_pattern': ['AU4', 'AU7', 'AU23', 'AU24'],
            'anxiety_pattern': ['AU1', 'AU2', 'AU5'],
            'suppression_pattern': ['AU15', 'AU17', 'AU24'],
            'concentration_pattern': ['AU4', 'AU7'],
            'frustration_pattern': ['AU4', 'AU9', 'AU10']
        }
        
        # Temporal analysis for micro-expressions
        self.micro_expression_threshold = 0.5  # seconds
        self.stress_history = []
        
    def analyze_stress_signals(self, action_units):
        """Analyze facial action units for stress indicators"""
        stress_indicators = {
            'micro_expressions': [],
            'tension_level': 0.0,
            'anxiety_level': 0.0,
            'overall_stress': 0.0,
            'detected_patterns': [],
            'confidence': 0.0
        }
        
        # Analyze individual action units
        total_stress_signal = 0.0
        detected_aus = 0
        
        for au_code, intensity in action_units.items():
            if au_code in self.stress_action_units and intensity > 0.3:
                weight = self.stress_action_units[au_code]['stress_weight']
                contribution = intensity * weight
                total_stress_signal += contribution
                detected_aus += 1
                
                # Check for micro-expressions (brief, intense activations)
                if intensity > 0.7:
                    stress_indicators['micro_expressions'].append({
                        'au': au_code,
                        'name': self.stress_action_units[au_code]['name'],
                        'intensity': intensity,
                        'timestamp': datetime.now()
                    })
        
        # Calculate overall stress level
        if detected_aus > 0:
            base_stress = total_stress_signal / detected_aus
            pattern_boost = len(stress_indicators['detected_patterns']) * 0.1
            stress_indicators['overall_stress'] = min(1.0, base_stress + pattern_boost)
            stress_indicators['confidence'] = min(1.0, detected_aus / 5.0)
        
        return stress_indicators
    
    def calculate_stress_score(self, stress_indicators):
        """Calculate normalized stress score (0-1 scale)"""
        base_score = stress_indicators['overall_stress']
        
        # Apply temporal smoothing
        current_time = datetime.now()
        self.stress_history.append({
            'score': base_score,
            'timestamp': current_time
        })
        
        # Keep only last 30 seconds of history
        cutoff_time = current_time - timedelta(seconds=30)
        self.stress_history = [h for h in self.stress_history if h['timestamp'] > cutoff_time]
        
        # Calculate smoothed score
        if len(self.stress_history) > 1:
            recent_scores = [h['score'] for h in self.stress_history[-10:]]
            smoothed_score = np.mean(recent_scores) * 0.7 + base_score * 0.3
        else:
            smoothed_score = base_score
        
        # Apply confidence weighting
        confidence = stress_indicators.get('confidence', 0.5)
        final_score = smoothed_score * confidence
        
        return round(final_score, 3)