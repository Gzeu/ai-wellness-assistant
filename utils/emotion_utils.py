"""
Emotion Analysis Utilities
Helper functions for emotion classification and interpretation
"""

import numpy as np
from datetime import datetime
from typing import Dict, List

class EmotionAnalyzer:
    def __init__(self):
        # Basic emotion categories with AU combinations
        self.emotion_patterns = {
            'happiness': {'aus': ['AU6', 'AU12'], 'threshold': 0.6},
            'sadness': {'aus': ['AU1', 'AU4', 'AU15'], 'threshold': 0.5},
            'anger': {'aus': ['AU4', 'AU5', 'AU7', 'AU23'], 'threshold': 0.6},
            'fear': {'aus': ['AU1', 'AU2', 'AU5', 'AU20'], 'threshold': 0.5},
            'surprise': {'aus': ['AU1', 'AU2', 'AU5', 'AU26'], 'threshold': 0.6},
            'disgust': {'aus': ['AU9', 'AU15', 'AU16'], 'threshold': 0.5},
            'contempt': {'aus': ['AU12', 'AU14'], 'threshold': 0.4}
        }
        
        # Stress-related emotional states
        self.stress_emotions = {
            'anxiety': ['fear', 'tension'],
            'frustration': ['anger', 'contempt'],
            'overwhelm': ['sadness', 'fear'],
            'suppression': ['contempt', 'disgust']
        }
        
    def classify_emotions(self, action_units: Dict[str, float]) -> Dict[str, float]:
        """Classify emotions based on action unit combinations"""
        emotion_scores = {}
        
        for emotion, pattern in self.emotion_patterns.items():
            score = self._calculate_emotion_score(action_units, pattern)
            if score >= pattern['threshold']:
                emotion_scores[emotion] = score
                
        return emotion_scores
    
    def _calculate_emotion_score(self, action_units: Dict[str, float], pattern: Dict) -> float:
        """Calculate emotion score based on AU pattern"""
        relevant_aus = pattern['aus']
        present_aus = 0
        total_intensity = 0.0
        
        for au in relevant_aus:
            if au in action_units and action_units[au] > 0.3:
                present_aus += 1
                total_intensity += action_units[au]
        
        if present_aus == 0:
            return 0.0
            
        coverage = present_aus / len(relevant_aus)
        average_intensity = total_intensity / present_aus
        
        return coverage * average_intensity
    
    def get_stress_related_emotions(self, emotions: Dict[str, float]) -> List[str]:
        """Identify stress-related emotions from emotion classification"""
        stress_related = []
        
        for stress_type, related_emotions in self.stress_emotions.items():
            for emotion in related_emotions:
                if emotion in emotions and emotions[emotion] > 0.5:
                    stress_related.append(stress_type)
                    break
                    
        return list(set(stress_related))
    
    def generate_emotion_summary(self, emotions: Dict[str, float], stress_score: float) -> str:
        """Generate human-readable emotion summary"""
        if not emotions:
            return "Neutral emotional state detected."
            
        primary_emotion = max(emotions, key=emotions.get)
        intensity = emotions[primary_emotion]
        
        intensity_desc = "slight" if intensity < 0.4 else "moderate" if intensity < 0.7 else "strong"
        
        summary = f"Primary emotion: {primary_emotion} ({intensity_desc} intensity)."
        
        if stress_score > 0.5:
            summary += f" Stress level: {stress_score:.1%}."
            
        return summary