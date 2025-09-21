"""
AI Wellness Coach - Personalized Stress Management and Relaxation Techniques
Provides evidence-based interventions based on detected stress patterns
"""

import random
from datetime import datetime, timedelta
import json

class WellnessCoach:
    def __init__(self):
        # Evidence-based relaxation techniques
        self.relaxation_techniques = {
            'breathing_exercises': {
                'box_breathing': {
                    'name': '4-7-8 Breathing Technique',
                    'description': 'Inhale for 4, hold for 7, exhale for 8 seconds',
                    'duration': '2-5 minutes',
                    'instructions': [
                        'Sit comfortably with your back straight',
                        'Place tongue behind upper teeth',
                        'Exhale completely through mouth', 
                        'Close mouth, inhale through nose for 4 counts',
                        'Hold breath for 7 counts',
                        'Exhale through mouth for 8 counts',
                        'Repeat 3-4 cycles'
                    ],
                    'stress_levels': ['moderate', 'high', 'very_high']
                },
                'belly_breathing': {
                    'name': 'Diaphragmatic Breathing',
                    'description': 'Deep breathing using your diaphragm',
                    'duration': '5-10 minutes',
                    'instructions': [
                        'Lie down or sit comfortably',
                        'Place one hand on chest, one on belly',
                        'Breathe slowly through nose',
                        'Feel belly rise more than chest',
                        'Exhale slowly through pursed lips',
                        'Continue for 5-10 minutes'
                    ],
                    'stress_levels': ['slight', 'moderate', 'high']
                }
            },
            'mindfulness_exercises': {
                '5_4_3_2_1_grounding': {
                    'name': '5-4-3-2-1 Grounding Technique',
                    'description': 'Use your senses to anchor in the present',
                    'duration': '3-5 minutes',
                    'instructions': [
                        'Notice 5 things you can see',
                        'Notice 4 things you can touch',
                        'Notice 3 things you can hear',
                        'Notice 2 things you can smell',
                        'Notice 1 thing you can taste',
                        'Take 3 deep breaths'
                    ],
                    'stress_levels': ['slight', 'moderate', 'high', 'very_high']
                }
            }
        }
        
        # Pattern-specific recommendations
        self.pattern_recommendations = {
            'tension_pattern': ['box_breathing', 'muscle_release'],
            'anxiety_pattern': ['belly_breathing', '5_4_3_2_1_grounding'],
            'suppression_pattern': ['belly_breathing', 'muscle_release']
        }
        
        # Encouragement messages
        self.encouragement_messages = {
            'relaxed': ["You're doing great! Your calm state is wonderful to see."],
            'moderate': ["I can see you're experiencing some stress. Let's work together."],
            'high': ["Significant stress detected. Let's focus on immediate relief."]
        }
        
        self.technique_history = []
        
    def get_stress_relief_techniques(self, stress_score, stress_indicators):
        """Get personalized stress relief recommendations"""
        stress_level = self._categorize_stress_level(stress_score)
        
        # Select techniques based on detected patterns
        recommended_techniques = []
        for pattern_info in stress_indicators.get('detected_patterns', []):
            pattern_name = pattern_info['pattern']
            if pattern_name in self.pattern_recommendations:
                recommended_techniques.extend(self.pattern_recommendations[pattern_name])
        
        # Get technique details
        techniques = []
        for technique_id in recommended_techniques[:3]:
            technique = self._find_technique_by_id(technique_id)
            if technique:
                techniques.append(technique)
        
        encouragement = random.choice(self.encouragement_messages.get(stress_level, ["You've got this!"]))
        
        return {
            'stress_level': stress_level,
            'stress_score': stress_score,
            'encouragement': encouragement,
            'immediate_techniques': techniques,
            'timestamp': datetime.now().isoformat()
        }
    
    def _categorize_stress_level(self, score):
        """Categorize stress score into level"""
        if score < 0.2:
            return 'relaxed'
        elif score < 0.4:
            return 'slight'
        elif score < 0.6:
            return 'moderate'
        elif score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _find_technique_by_id(self, technique_id):
        """Find technique details by ID"""
        for category, techniques in self.relaxation_techniques.items():
            if technique_id in techniques:
                technique = techniques[technique_id].copy()
                technique['category'] = category
                technique['id'] = technique_id
                return technique
        return None