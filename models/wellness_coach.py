"""
AI Wellness Coach - Personalized Stress Management and Relaxation Techniques
Provides evidence-based interventions based on detected stress patterns
Author: Pricop George - AI Wellness Assistant
"""

import random
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict, deque

class WellnessCoach:
    def __init__(self):
        # Comprehensive evidence-based relaxation techniques
        self.relaxation_techniques = {
            'breathing_exercises': {
                'box_breathing': {
                    'name': '4-7-8 Breathing Technique',
                    'description': 'Powerful breathing for rapid stress reduction',
                    'duration': '2-5 minutes',
                    'effectiveness': 0.85,
                    'instructions': [
                        'Sit comfortably with your back straight',
                        'Place tongue behind upper teeth',
                        'Exhale completely through mouth', 
                        'Close mouth, inhale through nose for 4 counts',
                        'Hold breath for 7 counts',
                        'Exhale through mouth for 8 counts',
                        'Repeat 3-4 cycles'
                    ],
                    'stress_patterns': ['tension', 'anxiety', 'frustration'],
                    'stress_levels': ['moderate', 'high', 'very_high'],
                    'benefits': 'Reduces cortisol, activates parasympathetic nervous system'
                },
                'belly_breathing': {
                    'name': 'Diaphragmatic Breathing',
                    'description': 'Deep abdominal breathing for sustained calm',
                    'duration': '5-10 minutes',
                    'effectiveness': 0.78,
                    'instructions': [
                        'Lie down or sit comfortably',
                        'Place one hand on chest, one on belly',
                        'Breathe slowly through nose',
                        'Feel belly rise more than chest',
                        'Exhale slowly through pursed lips',
                        'Continue for 5-10 minutes'
                    ],
                    'stress_patterns': ['anxiety', 'suppression'],
                    'stress_levels': ['slight', 'moderate', 'high'],
                    'benefits': 'Lowers blood pressure, improves focus'
                },
                'coherent_breathing': {
                    'name': 'Coherent Breathing 5-5',
                    'description': '5 seconds in, 5 seconds out for balance',
                    'duration': '3-10 minutes',
                    'effectiveness': 0.72,
                    'instructions': [
                        'Sit or lie in comfortable position',
                        'Breathe in slowly for 5 counts',
                        'Breathe out slowly for 5 counts',
                        'Keep rhythm steady and relaxed',
                        'Focus on smooth, even breaths'
                    ],
                    'stress_patterns': ['tension', 'concentration'],
                    'stress_levels': ['slight', 'moderate'],
                    'benefits': 'Heart rate variability optimization'
                }
            },
            'progressive_relaxation': {
                'muscle_relaxation': {
                    'name': 'Progressive Muscle Relaxation',
                    'description': 'Systematic tension and release of muscle groups',
                    'duration': '10-20 minutes',
                    'effectiveness': 0.82,
                    'instructions': [
                        'Start with your toes, tense for 5 seconds',
                        'Release tension, notice the relaxation',
                        'Move up to calves, thighs, abdomen',
                        'Continue to arms, shoulders, neck, face',
                        'End with whole-body tension and release',
                        'Rest in complete relaxation for 2 minutes'
                    ],
                    'stress_patterns': ['tension', 'suppression'],
                    'stress_levels': ['high', 'very_high'],
                    'benefits': 'Reduces muscle tension, improves sleep'
                },
                'facial_release': {
                    'name': 'Facial Muscle Release',
                    'description': 'Targeted relaxation for facial tension',
                    'duration': '2-5 minutes',
                    'effectiveness': 0.70,
                    'instructions': [
                        'Close eyes gently',
                        'Relax forehead and eyebrows',
                        'Unclench jaw, let it drop slightly',
                        'Relax lips and cheek muscles',
                        'Breathe naturally for 30 seconds',
                        'Repeat 3-5 times'
                    ],
                    'stress_patterns': ['tension', 'frustration'],
                    'stress_levels': ['moderate', 'high'],
                    'benefits': 'Reduces facial tension, prevents headaches'
                }
            },
            'mindfulness_techniques': {
                '5_4_3_2_1_grounding': {
                    'name': '5-4-3-2-1 Grounding Technique',
                    'description': 'Sensory anchoring to the present moment',
                    'duration': '3-5 minutes',
                    'effectiveness': 0.75,
                    'instructions': [
                        'Notice 5 things you can see',
                        'Notice 4 things you can touch',
                        'Notice 3 things you can hear',
                        'Notice 2 things you can smell',
                        'Notice 1 thing you can taste',
                        'Take 3 deep, conscious breaths'
                    ],
                    'stress_patterns': ['anxiety', 'suppression', 'frustration'],
                    'stress_levels': ['slight', 'moderate', 'high', 'very_high'],
                    'benefits': 'Interrupts anxiety spirals, grounds attention'
                },
                'body_scan': {
                    'name': 'Body Scan Meditation',
                    'description': 'Mindful attention to physical sensations',
                    'duration': '5-15 minutes',
                    'effectiveness': 0.80,
                    'instructions': [
                        'Lie down or sit comfortably',
                        'Close eyes, focus on breathing',
                        'Start attention at top of head',
                        'Slowly move attention down body',
                        'Notice sensations without judgment',
                        'Finish at toes, rest in awareness'
                    ],
                    'stress_patterns': ['tension', 'anxiety', 'suppression'],
                    'stress_levels': ['moderate', 'high', 'very_high'],
                    'benefits': 'Increases body awareness, reduces rumination'
                }
            },
            'movement_therapy': {
                'neck_stretching': {
                    'name': 'Neck and Shoulder Release',
                    'description': 'Gentle movements to release upper body tension',
                    'duration': '2-5 minutes',
                    'effectiveness': 0.68,
                    'instructions': [
                        'Sit tall, shoulders relaxed',
                        'Slowly turn head left, hold 10 seconds',
                        'Return center, turn right, hold 10 seconds',
                        'Tilt head to left shoulder, hold 10 seconds',
                        'Tilt to right shoulder, hold 10 seconds',
                        'Roll shoulders backward 5 times'
                    ],
                    'stress_patterns': ['tension'],
                    'stress_levels': ['moderate', 'high'],
                    'benefits': 'Releases muscle tension, improves circulation'
                }
            }
        }
        
        # Enhanced pattern-to-technique mapping
        self.pattern_recommendations = {
            'tension': {
                'primary': ['box_breathing', 'muscle_relaxation', 'facial_release'],
                'secondary': ['neck_stretching', 'coherent_breathing'],
                'urgency': 'high'
            },
            'anxiety': {
                'primary': ['belly_breathing', '5_4_3_2_1_grounding'],
                'secondary': ['body_scan', 'coherent_breathing'],
                'urgency': 'high'
            },
            'suppression': {
                'primary': ['belly_breathing', 'muscle_relaxation'],
                'secondary': ['body_scan', '5_4_3_2_1_grounding'],
                'urgency': 'medium'
            },
            'concentration': {
                'primary': ['coherent_breathing', 'neck_stretching'],
                'secondary': ['belly_breathing'],
                'urgency': 'low'
            },
            'frustration': {
                'primary': ['box_breathing', '5_4_3_2_1_grounding'],
                'secondary': ['facial_release', 'muscle_relaxation'],
                'urgency': 'high'
            }
        }
        
        # Contextual encouragement messages
        self.encouragement_messages = {
            'low': [
                "Você está em um ótimo estado! Continue assim! 🌟",
                "Seu nível de calma é inspirador. Mantenha essa energia! ✨",
                "Estado relaxado detectado. Aproveite esse momento de paz! 🕊️"
            ],
            'moderate': [
                "Percebo um pouco de tensão. Vamos trabalhar juntos para aliviá-la! 🤝",
                "Algumas técnicas de respiração podem ser úteis agora. 🌬️",
                "Você está lidando bem, mas posso ajudar a melhorar ainda mais! 💪"
            ],
            'high': [
                "Detectei níveis altos de stress. Vou guiá-lo para o alívio imediato! 🆘",
                "Momento de parar e respirar. Você consegue superar isso! 🛡️",
                "Estou aqui para ajudar. Vamos focar nas técnicas mais eficazes! 🎯"
            ],
            'very_high': [
                "Stress muito alto detectado. Prioridade máxima no seu bem-estar! 🚨",
                "Hora de cuidar de você imediatamente. Siga comigo passo a passo! ⚡",
                "Situação crítica - vamos aplicar as técnicas mais poderosas agora! 💥"
            ]
        }
        
        # User progress tracking
        self.technique_usage = defaultdict(int)
        self.technique_effectiveness = defaultdict(list)
        self.user_preferences = {
            'preferred_duration': 'medium',  # short, medium, long
            'preferred_categories': [],
            'effectiveness_threshold': 0.7
        }
        
        # Session tracking for adaptive recommendations
        self.session_history = deque(maxlen=100)
        
    def get_recommendations(self, stress_score, stress_indicators=None, user_context=None):
        """Get comprehensive wellness recommendations based on current state"""
        if stress_indicators is None:
            stress_indicators = {'detected_patterns': {}}
            
        stress_level = self._categorize_stress_level(stress_score)
        
        # Identify primary patterns for targeted recommendations
        detected_patterns = stress_indicators.get('detected_patterns', {})
        
        # Get pattern-specific recommendations
        recommended_techniques = self._get_pattern_specific_techniques(detected_patterns, stress_level)
        
        # Apply personalization based on history
        personalized_techniques = self._personalize_recommendations(recommended_techniques, stress_level)
        
        # Generate encouragement message
        encouragement = self._get_contextual_encouragement(stress_level, detected_patterns)
        
        # Create comprehensive response
        recommendations = {
            'stress_assessment': {
                'level': stress_level,
                'score': stress_score,
                'dominant_pattern': stress_indicators.get('dominant_pattern'),
                'confidence': stress_indicators.get('confidence', 0.0)
            },
            'immediate_actions': personalized_techniques[:2],  # Top 2 recommendations
            'additional_techniques': personalized_techniques[2:5],  # Next 3 options
            'encouragement': encouragement,
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'personalization_active': len(self.technique_effectiveness) > 0,
                'session_count': len(self.session_history)
            },
            'insights': self._generate_insights(stress_indicators)
        }
        
        # Record session for learning
        self._record_session(stress_score, detected_patterns, personalized_techniques)
        
        return recommendations
    
    def _get_pattern_specific_techniques(self, detected_patterns, stress_level):
        """Get techniques specific to detected stress patterns"""
        technique_scores = defaultdict(float)
        
        # Score techniques based on detected patterns
        for pattern_name, pattern_data in detected_patterns.items():
            if pattern_name in self.pattern_recommendations:
                pattern_score = pattern_data.get('score', 0.5)
                urgency = self.pattern_recommendations[pattern_name]['urgency']
                
                # Primary techniques get higher scores
                for technique_id in self.pattern_recommendations[pattern_name]['primary']:
                    urgency_multiplier = {'high': 1.0, 'medium': 0.8, 'low': 0.6}[urgency]
                    technique_scores[technique_id] += pattern_score * urgency_multiplier
                    
                # Secondary techniques get lower scores
                for technique_id in self.pattern_recommendations[pattern_name]['secondary']:
                    urgency_multiplier = {'high': 0.6, 'medium': 0.5, 'low': 0.4}[urgency]
                    technique_scores[technique_id] += pattern_score * urgency_multiplier * 0.7
        
        # If no patterns detected, use general stress-level recommendations
        if not technique_scores:
            technique_scores = self._get_general_recommendations(stress_level)
        
        # Sort techniques by score and get details
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommended_techniques = []
        for technique_id, score in sorted_techniques:
            technique = self._find_technique_by_id(technique_id)
            if technique:
                technique['recommendation_score'] = score
                recommended_techniques.append(technique)
        
        return recommended_techniques
    
    def _personalize_recommendations(self, techniques, stress_level):
        """Personalize recommendations based on user history and preferences"""
        if not self.technique_effectiveness:
            return techniques  # No history yet
        
        # Adjust scores based on personal effectiveness
        for technique in techniques:
            technique_id = technique['id']
            
            # Historical effectiveness
            if technique_id in self.technique_effectiveness:
                personal_effectiveness = np.mean(self.technique_effectiveness[technique_id])
                # Boost score for personally effective techniques
                technique['recommendation_score'] *= (0.5 + personal_effectiveness)
                technique['personal_effectiveness'] = personal_effectiveness
            
            # Usage frequency (variety is good)
            usage_count = self.technique_usage[technique_id]
            total_usage = sum(self.technique_usage.values())
            if total_usage > 0:
                usage_ratio = usage_count / total_usage
                # Slightly penalize overused techniques to encourage variety
                if usage_ratio > 0.4:
                    technique['recommendation_score'] *= 0.9
        
        # Re-sort based on personalized scores
        techniques.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return techniques
    
    def _get_general_recommendations(self, stress_level):
        """Get general recommendations when no specific patterns detected"""
        general_scores = {}
        
        for category, techniques in self.relaxation_techniques.items():
            for technique_id, technique_info in techniques.items():
                if stress_level in technique_info['stress_levels']:
                    # Score based on effectiveness and stress level match
                    base_score = technique_info['effectiveness']
                    level_match = 1.0 if stress_level in technique_info['stress_levels'] else 0.5
                    general_scores[technique_id] = base_score * level_match
        
        return general_scores
    
    def _get_contextual_encouragement(self, stress_level, detected_patterns):
        """Generate contextual encouragement based on current state"""
        base_messages = self.encouragement_messages.get(stress_level, [])
        
        # Add pattern-specific context
        if detected_patterns:
            pattern_names = list(detected_patterns.keys())
            if 'tension' in pattern_names:
                base_messages.extend(["Percebo tensão muscular. Vamos aliviar isso juntos! 💆"])
            if 'anxiety' in pattern_names:
                base_messages.extend(["Ansiedade detectada. Respiração profunda será sua aliada! 🌬️"])
            if 'frustration' in pattern_names:
                base_messages.extend(["Frustração é natural. Vamos canalizar essa energia positivamente! ⚡"])
        
        return random.choice(base_messages) if base_messages else "Estou aqui para ajudar! 🤗"
    
    def _generate_insights(self, stress_indicators):
        """Generate actionable insights from stress analysis"""
        insights = []
        
        # Pattern-specific insights
        detected_patterns = stress_indicators.get('detected_patterns', {})
        for pattern_name, pattern_data in detected_patterns.items():
            intensity = pattern_data.get('intensity', 'moderate')
            insights.append(f"{pattern_name.title()}: {pattern_data['description']} (Intensidade: {intensity})")
        
        # Micro-expression insights
        micro_expressions = stress_indicators.get('micro_expressions', [])
        if micro_expressions:
            insights.append(f"Micro-expressões detectadas: {len(micro_expressions)} indicadores rápidos")
        
        # Confidence insights
        confidence = stress_indicators.get('confidence', 0.0)
        if confidence > 0.8:
            insights.append("Análise muito confiável - recomendações precisas")
        elif confidence < 0.5:
            insights.append("Confiança moderada - considere melhor iluminação")
        
        return insights
    
    def record_technique_feedback(self, technique_id, effectiveness_rating, duration_used=None):
        """Record user feedback on technique effectiveness"""
        self.technique_usage[technique_id] += 1
        self.technique_effectiveness[technique_id].append(effectiveness_rating)
        
        # Keep only recent effectiveness ratings
        if len(self.technique_effectiveness[technique_id]) > 20:
            self.technique_effectiveness[technique_id].pop(0)
        
        # Update user preferences based on feedback
        if effectiveness_rating >= 0.8:
            technique = self._find_technique_by_id(technique_id)
            if technique:
                category = technique['category']
                if category not in self.user_preferences['preferred_categories']:
                    self.user_preferences['preferred_categories'].append(category)
    
    def _record_session(self, stress_score, patterns, recommended_techniques):
        """Record session data for adaptive learning"""
        session_data = {
            'timestamp': datetime.now(),
            'stress_score': stress_score,
            'patterns': list(patterns.keys()) if patterns else [],
            'recommended_techniques': [t['id'] for t in recommended_techniques[:3]],
            'stress_category': self._categorize_stress_level(stress_score)
        }
        
        self.session_history.append(session_data)
    
    def get_progress_insights(self, days_back=7):
        """Generate progress insights from recent sessions"""
        if not self.session_history:
            return {'status': 'no_data'}
        
        # Filter recent sessions
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_sessions = [s for s in self.session_history if s['timestamp'] > cutoff_date]
        
        if not recent_sessions:
            return {'status': 'no_recent_data'}
        
        # Calculate progress metrics
        stress_scores = [s['stress_score'] for s in recent_sessions]
        
        insights = {
            'total_sessions': len(recent_sessions),
            'average_stress': np.mean(stress_scores),
            'stress_trend': 'improving' if len(stress_scores) > 1 and stress_scores[-1] < stress_scores[0] else 'stable',
            'most_common_pattern': self._get_most_common_pattern(recent_sessions),
            'most_effective_techniques': self._get_most_effective_techniques(),
            'recommendations_for_improvement': self._generate_improvement_recommendations(recent_sessions)
        }
        
        return insights
    
    def _get_most_common_pattern(self, sessions):
        """Identify most frequently occurring stress pattern"""
        pattern_counts = defaultdict(int)
        for session in sessions:
            for pattern in session['patterns']:
                pattern_counts[pattern] += 1
        
        if not pattern_counts:
            return None
            
        return max(pattern_counts.items(), key=lambda x: x[1])
    
    def _get_most_effective_techniques(self, limit=3):
        """Get most effective techniques based on user feedback"""
        if not self.technique_effectiveness:
            return []
        
        technique_avg_effectiveness = {}
        for technique_id, ratings in self.technique_effectiveness.items():
            if len(ratings) >= 2:  # Need minimum ratings for reliability
                technique_avg_effectiveness[technique_id] = {
                    'average_effectiveness': np.mean(ratings),
                    'usage_count': len(ratings),
                    'name': self._find_technique_by_id(technique_id)['name'] if self._find_technique_by_id(technique_id) else technique_id
                }
        
        # Sort by effectiveness
        sorted_techniques = sorted(technique_avg_effectiveness.items(), 
                                 key=lambda x: x[1]['average_effectiveness'], reverse=True)
        
        return sorted_techniques[:limit]
    
    def _generate_improvement_recommendations(self, recent_sessions):
        """Generate recommendations for improvement based on patterns"""
        recommendations = []
        
        # Check for persistent high stress
        high_stress_sessions = [s for s in recent_sessions if s['stress_score'] > 0.6]
        if len(high_stress_sessions) > len(recent_sessions) * 0.6:
            recommendations.append("Considere práticas regulares de mindfulness para reduzir stress de base")
        
        # Check for pattern consistency
        all_patterns = [p for s in recent_sessions for p in s['patterns']]
        if all_patterns and all_patterns.count(max(set(all_patterns), key=all_patterns.count)) > len(all_patterns) * 0.4:
            dominant_pattern = max(set(all_patterns), key=all_patterns.count)
            recommendations.append(f"Padrão {dominant_pattern} recorrente - explore técnicas preventivas")
        
        # Check technique variety
        all_techniques = [t for s in recent_sessions for t in s['recommended_techniques']]
        unique_techniques = set(all_techniques)
        if len(unique_techniques) < 3:
            recommendations.append("Experimente maior variedade de técnicas para melhor eficácia")
        
        return recommendations
    
    def _categorize_stress_level(self, score):
        """Categorize stress score into descriptive level"""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'moderate'
        elif score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _find_technique_by_id(self, technique_id):
        """Find technique details by ID across all categories"""
        for category_name, techniques in self.relaxation_techniques.items():
            if technique_id in techniques:
                technique = techniques[technique_id].copy()
                technique['category'] = category_name
                technique['id'] = technique_id
                return technique
        return None
    
    def export_user_data(self):
        """Export user progress data for backup or analysis"""
        return {
            'technique_usage': dict(self.technique_usage),
            'technique_effectiveness': {k: list(v) for k, v in self.technique_effectiveness.items()},
            'user_preferences': self.user_preferences,
            'session_count': len(self.session_history),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_user_data(self, data):
        """Import user progress data from backup"""
        if 'technique_usage' in data:
            self.technique_usage.update(data['technique_usage'])
        if 'technique_effectiveness' in data:
            for k, v in data['technique_effectiveness'].items():
                self.technique_effectiveness[k].extend(v)
        if 'user_preferences' in data:
            self.user_preferences.update(data['user_preferences'])