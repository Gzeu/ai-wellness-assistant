"""
Advanced Stress Detection using Facial Action Coding System (FACS)
Implements real-time stress analysis from micro-expressions with pattern recognition
Author: Pricop George - AI Wellness Assistant
"""

import numpy as np
from datetime import datetime, timedelta
import math
from collections import deque, defaultdict

class StressDetector:
    def __init__(self):
        # Enhanced FACS Action Units with refined stress weights
        self.stress_action_units = {
            'AU1': {'name': 'Inner Brow Raiser', 'stress_weight': 0.7, 'reliability': 0.85},
            'AU2': {'name': 'Outer Brow Raiser', 'stress_weight': 0.6, 'reliability': 0.80},
            'AU4': {'name': 'Brow Lowerer', 'stress_weight': 0.9, 'reliability': 0.90},  # Strong stress indicator
            'AU5': {'name': 'Upper Lid Raiser', 'stress_weight': 0.5, 'reliability': 0.75},
            'AU7': {'name': 'Lid Tightener', 'stress_weight': 0.8, 'reliability': 0.88},  # High stress correlation
            'AU9': {'name': 'Nose Wrinkler', 'stress_weight': 0.4, 'reliability': 0.70},
            'AU10': {'name': 'Upper Lip Raiser', 'stress_weight': 0.3, 'reliability': 0.65},
            'AU11': {'name': 'Nasolabial Deepener', 'stress_weight': 0.4, 'reliability': 0.72},
            'AU14': {'name': 'Dimpler', 'stress_weight': 0.5, 'reliability': 0.76},
            'AU15': {'name': 'Lip Corner Depressor', 'stress_weight': 0.7, 'reliability': 0.82},
            'AU17': {'name': 'Chin Raiser', 'stress_weight': 0.4, 'reliability': 0.68},
            'AU20': {'name': 'Lip Stretcher', 'stress_weight': 0.6, 'reliability': 0.78},
            'AU22': {'name': 'Lip Funneler', 'stress_weight': 0.3, 'reliability': 0.62},
            'AU23': {'name': 'Lip Tightener', 'stress_weight': 0.8, 'reliability': 0.85},  # Strong indicator
            'AU24': {'name': 'Lip Pressor', 'stress_weight': 0.7, 'reliability': 0.83},
        }
        
        # Advanced stress pattern recognition with weighted combinations
        self.stress_patterns = {
            'tension': {
                'aus': ['AU4', 'AU7', 'AU23', 'AU24'],
                'weight': 0.9,  # High reliability for tension detection
                'threshold': 0.6,
                'description': 'Facial tension and muscle strain'
            },
            'anxiety': {
                'aus': ['AU1', 'AU2', 'AU5'],
                'weight': 0.8,
                'threshold': 0.5,
                'description': 'Anxiety and worry indicators'
            },
            'suppression': {
                'aus': ['AU15', 'AU17', 'AU24'],
                'weight': 0.7,
                'threshold': 0.6,
                'description': 'Emotional suppression and control'
            },
            'concentration': {
                'aus': ['AU4', 'AU7'],
                'weight': 0.6,
                'threshold': 0.7,
                'description': 'Intense concentration and focus'
            },
            'frustration': {
                'aus': ['AU4', 'AU9', 'AU10'],
                'weight': 0.8,
                'threshold': 0.5,
                'description': 'Frustration and irritation'
            }
        }
        
        # Temporal analysis parameters
        self.micro_expression_threshold = 0.5  # seconds for micro-expression detection
        self.stress_history = deque(maxlen=100)  # Efficient circular buffer
        self.pattern_history = defaultdict(list)
        self.confidence_history = deque(maxlen=50)
        
        # Adaptive parameters
        self.baseline_stress = 0.0
        self.stress_sensitivity = 1.0
        self.temporal_weight = 0.3  # Weight for temporal smoothing
        
    def analyze_stress_signals(self, action_units, timestamp=None):
        """Enhanced stress signal analysis with pattern recognition"""
        if timestamp is None:
            timestamp = datetime.now()
            
        stress_indicators = {
            'micro_expressions': [],
            'detected_patterns': {},
            'pattern_scores': {},
            'overall_stress': 0.0,
            'confidence': 0.0,
            'stress_level': 'low',
            'dominant_pattern': None,
            'reliability_score': 0.0
        }
        
        # Analyze individual action units for micro-expressions
        total_stress_signal = 0.0
        total_reliability = 0.0
        detected_aus = 0
        
        for au_code, intensity in action_units.items():
            if au_code in self.stress_action_units and intensity > 0.2:  # Lower threshold for detection
                au_info = self.stress_action_units[au_code]
                weight = au_info['stress_weight']
                reliability = au_info['reliability']
                
                # Weighted contribution to stress signal
                contribution = intensity * weight * reliability
                total_stress_signal += contribution
                total_reliability += reliability
                detected_aus += 1
                
                # Detect micro-expressions (brief, intense activations)
                if intensity > 0.75:  # High intensity threshold
                    stress_indicators['micro_expressions'].append({
                        'au': au_code,
                        'name': au_info['name'],
                        'intensity': intensity,
                        'reliability': reliability,
                        'timestamp': timestamp
                    })
        
        # Calculate pattern-specific stress levels
        pattern_contributions = []
        for pattern_name, pattern_info in self.stress_patterns.items():
            pattern_score = self._calculate_pattern_score(action_units, pattern_info)
            stress_indicators['pattern_scores'][pattern_name] = pattern_score
            
            if pattern_score > pattern_info['threshold']:
                stress_indicators['detected_patterns'][pattern_name] = {
                    'score': pattern_score,
                    'description': pattern_info['description'],
                    'weight': pattern_info['weight'],
                    'intensity': 'high' if pattern_score > 0.8 else 'moderate'
                }
                pattern_contributions.append(pattern_score * pattern_info['weight'])
        
        # Calculate overall stress level
        if detected_aus > 0:
            # Base stress from individual AUs
            base_stress = total_stress_signal / max(detected_aus, 1)
            
            # Pattern boost from detected stress patterns
            pattern_boost = np.mean(pattern_contributions) if pattern_contributions else 0.0
            
            # Combined stress score with pattern weighting
            combined_stress = (base_stress * 0.6) + (pattern_boost * 0.4)
            stress_indicators['overall_stress'] = min(1.0, combined_stress)
            
            # Enhanced confidence calculation
            base_confidence = min(1.0, detected_aus / 8.0)  # Based on number of detected AUs
            reliability_confidence = total_reliability / max(detected_aus, 1)
            pattern_confidence = len(stress_indicators['detected_patterns']) / 5.0
            
            stress_indicators['confidence'] = min(1.0, 
                (base_confidence * 0.4) + 
                (reliability_confidence * 0.4) + 
                (pattern_confidence * 0.2)
            )
            
            stress_indicators['reliability_score'] = reliability_confidence
        
        # Determine dominant pattern
        if stress_indicators['detected_patterns']:
            dominant = max(stress_indicators['detected_patterns'].items(), 
                          key=lambda x: x[1]['score'])
            stress_indicators['dominant_pattern'] = {
                'name': dominant[0],
                'score': dominant[1]['score'],
                'description': dominant[1]['description']
            }
        
        return stress_indicators
    
    def _calculate_pattern_score(self, action_units, pattern_info):
        """Calculate stress score for a specific pattern"""
        pattern_aus = pattern_info['aus']
        active_intensities = []
        
        for au in pattern_aus:
            if au in action_units and action_units[au] > 0.2:
                # Weight by AU reliability
                reliability = self.stress_action_units.get(au, {}).get('reliability', 0.7)
                weighted_intensity = action_units[au] * reliability
                active_intensities.append(weighted_intensity)
        
        if not active_intensities:
            return 0.0
        
        # Calculate pattern score based on average intensity and coverage
        avg_intensity = np.mean(active_intensities)
        coverage_ratio = len(active_intensities) / len(pattern_aus)
        
        # Combine intensity and coverage for final pattern score
        pattern_score = avg_intensity * (0.7 + 0.3 * coverage_ratio)
        
        return min(1.0, pattern_score)
    
    def calculate_stress_score(self, stress_indicators, apply_smoothing=True):
        """Calculate normalized stress score with temporal smoothing and confidence weighting"""
        base_score = stress_indicators['overall_stress']
        confidence = stress_indicators['confidence']
        timestamp = datetime.now()
        
        # Store current measurement
        measurement = {
            'score': base_score,
            'confidence': confidence,
            'timestamp': timestamp,
            'patterns': list(stress_indicators['detected_patterns'].keys())
        }
        
        self.stress_history.append(measurement)
        self.confidence_history.append(confidence)
        
        if not apply_smoothing or len(self.stress_history) < 3:
            final_score = base_score * confidence
        else:
            # Advanced temporal smoothing with confidence weighting
            recent_measurements = list(self.stress_history)[-10:]  # Last 10 measurements
            
            # Weight recent scores by their confidence and recency
            weighted_scores = []
            total_weight = 0.0
            
            for i, measurement in enumerate(recent_measurements):
                # Recency weight (more recent = higher weight)
                recency_weight = (i + 1) / len(recent_measurements)
                # Confidence weight
                conf_weight = measurement['confidence']
                # Combined weight
                combined_weight = recency_weight * conf_weight
                
                weighted_scores.append(measurement['score'] * combined_weight)
                total_weight += combined_weight
            
            if total_weight > 0:
                smoothed_score = sum(weighted_scores) / total_weight
                # Blend smoothed score with current score
                final_score = (smoothed_score * self.temporal_weight + 
                              base_score * (1 - self.temporal_weight))
            else:
                final_score = base_score
            
            # Apply current confidence weighting
            final_score *= confidence
        
        return round(min(1.0, max(0.0, final_score)), 3)
    
    def get_stress_interpretation(self, stress_score, stress_indicators):
        """Provide human-readable interpretation of stress levels"""
        # Determine stress level category
        if stress_score < 0.3:
            level = 'low'
            description = 'Minimal stress detected - relaxed state'
            color = '#4CAF50'  # Green
        elif stress_score < 0.6:
            level = 'moderate' 
            description = 'Moderate stress levels - some tension present'
            color = '#FF9800'  # Orange
        elif stress_score < 0.8:
            level = 'high'
            description = 'High stress detected - significant tension'
            color = '#FF5722'  # Deep Orange
        else:
            level = 'very_high'
            description = 'Very high stress - immediate attention recommended'
            color = '#F44336'  # Red
        
        # Generate pattern-specific insights
        insights = []
        if stress_indicators['detected_patterns']:
            for pattern_name, pattern_data in stress_indicators['detected_patterns'].items():
                insights.append(f"{pattern_name.title()}: {pattern_data['description']} "
                              f"(Score: {pattern_data['score']:.2f})")
        
        # Confidence assessment
        confidence = stress_indicators['confidence']
        if confidence > 0.8:
            confidence_text = 'Very Reliable'
        elif confidence > 0.6:
            confidence_text = 'Reliable'
        elif confidence > 0.4:
            confidence_text = 'Moderate Reliability'
        else:
            confidence_text = 'Low Reliability'
        
        return {
            'level': level,
            'score': stress_score,
            'description': description,
            'color': color,
            'confidence': confidence,
            'confidence_text': confidence_text,
            'insights': insights,
            'dominant_pattern': stress_indicators.get('dominant_pattern'),
            'micro_expressions_detected': len(stress_indicators['micro_expressions']),
            'patterns_detected': len(stress_indicators['detected_patterns'])
        }
    
    def reset_baseline(self):
        """Reset baseline measurements for personalized adaptation"""
        self.stress_history.clear()
        self.confidence_history.clear()
        self.pattern_history.clear()
        self.baseline_stress = 0.0
        
    def get_performance_metrics(self):
        """Get detector performance metrics for monitoring"""
        if not self.stress_history:
            return {'status': 'no_data'}
        
        recent_scores = [m['score'] for m in self.stress_history]
        recent_confidences = [m['confidence'] for m in self.stress_history]
        
        return {
            'measurements_count': len(self.stress_history),
            'average_score': np.mean(recent_scores),
            'score_variance': np.var(recent_scores),
            'average_confidence': np.mean(recent_confidences),
            'detection_stability': 1.0 - min(1.0, np.var(recent_scores) * 2),
            'last_measurement_age': (datetime.now() - self.stress_history[-1]['timestamp']).total_seconds()
        }