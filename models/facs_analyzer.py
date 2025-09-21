"""
Facial Action Coding System (FACS) Analyzer
Detects and quantifies facial action units for emotion and stress analysis
Based on Ekman & Friesen FACS methodology with MediaPipe integration
Author: Pricop George - AI Wellness Assistant
"""

import numpy as np
import math
from datetime import datetime, timedelta
from collections import deque
import cv2

class FACSAnalyzer:
    def __init__(self):
        # Enhanced FACS Action Unit definitions with stress relevance
        self.action_unit_definitions = {
            'AU1': {  # Inner Brow Raiser - STRESS PRIORITY
                'name': 'Inner Brow Raiser',
                'landmarks': [17, 18, 19, 20],  # Inner eyebrow points
                'description': 'Raises inner portion of eyebrows - worry/concern',
                'muscles': ['frontalis_medial'],
                'stress_relevance': 0.85,
                'calculation_method': 'vertical_displacement'
            },
            'AU2': {  # Outer Brow Raiser
                'name': 'Outer Brow Raiser', 
                'landmarks': [21, 22, 23, 24, 25, 26],
                'description': 'Raises outer portion of eyebrows - surprise/fear',
                'muscles': ['frontalis_lateral'],
                'stress_relevance': 0.70,
                'calculation_method': 'vertical_displacement'
            },
            'AU4': {  # Brow Lowerer - STRESS PRIORITY
                'name': 'Brow Lowerer',
                'landmarks': [17, 18, 19, 20, 21, 22],
                'description': 'Lowers and draws eyebrows together - concentration/tension',
                'muscles': ['corrugator', 'procerus'],
                'stress_relevance': 0.95,  # Highest stress indicator
                'calculation_method': 'vertical_displacement_and_convergence'
            },
            'AU5': {  # Upper Lid Raiser
                'name': 'Upper Lid Raiser',
                'landmarks': [37, 38, 40, 41, 43, 44, 46, 47],
                'description': 'Widens eye aperture - surprise/fear',
                'muscles': ['levator_palpebrae_superioris'],
                'stress_relevance': 0.60,
                'calculation_method': 'vertical_eye_opening'
            },
            'AU7': {  # Lid Tightener - STRESS PRIORITY  
                'name': 'Lid Tightener',
                'landmarks': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                'description': 'Tightens eyelids - tension/squinting',
                'muscles': ['orbicularis_oculi'],
                'stress_relevance': 0.88,
                'calculation_method': 'eye_area_reduction'
            },
            'AU9': {  # Nose Wrinkler
                'name': 'Nose Wrinkler',
                'landmarks': [31, 32, 33, 34, 35],
                'description': 'Wrinkles nose bridge - disgust/frustration',
                'muscles': ['levator_labii_superioris_alaeque_nasi'],
                'stress_relevance': 0.65,
                'calculation_method': 'nostril_displacement'
            },
            'AU10': {  # Upper Lip Raiser
                'name': 'Upper Lip Raiser',
                'landmarks': [48, 49, 50, 51, 52, 53, 54],
                'description': 'Raises upper lip - disgust/contempt',
                'muscles': ['levator_labii_superioris'],
                'stress_relevance': 0.55,
                'calculation_method': 'lip_elevation'
            },
            'AU11': {  # Nasolabial Deepener
                'name': 'Nasolabial Deepener',
                'landmarks': [31, 48, 49, 50],
                'description': 'Deepens nasolabial furrow - sadness/pain',
                'muscles': ['levator_labii_superioris'],
                'stress_relevance': 0.68,
                'calculation_method': 'furrow_depth'
            },
            'AU14': {  # Dimpler
                'name': 'Dimpler',
                'landmarks': [48, 54, 57, 58],
                'description': 'Pulls lip corners laterally - controlled smile/tension',
                'muscles': ['buccinator'],
                'stress_relevance': 0.58,
                'calculation_method': 'lateral_lip_pull'
            },
            'AU15': {  # Lip Corner Depressor - STRESS PRIORITY
                'name': 'Lip Corner Depressor',
                'landmarks': [48, 54],
                'description': 'Pulls lip corners downward - sadness/disappointment',
                'muscles': ['depressor_anguli_oris'],
                'stress_relevance': 0.82,
                'calculation_method': 'corner_depression'
            },
            'AU17': {  # Chin Raiser
                'name': 'Chin Raiser',
                'landmarks': [6, 7, 8, 9, 10],
                'description': 'Raises and protrudes chin - doubt/defiance',
                'muscles': ['mentalis'],
                'stress_relevance': 0.62,
                'calculation_method': 'chin_protrusion'
            },
            'AU20': {  # Lip Stretcher
                'name': 'Lip Stretcher',
                'landmarks': [48, 54, 60, 64],
                'description': 'Stretches lips horizontally - fear/tension',
                'muscles': ['risorius'],
                'stress_relevance': 0.72,
                'calculation_method': 'horizontal_lip_stretch'
            },
            'AU23': {  # Lip Tightener - STRESS PRIORITY
                'name': 'Lip Tightener',
                'landmarks': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                'description': 'Tightens and narrows lips - tension/concentration',
                'muscles': ['orbicularis_oris'],
                'stress_relevance': 0.85,
                'calculation_method': 'lip_compression'
            },
            'AU24': {  # Lip Pressor
                'name': 'Lip Pressor',
                'landmarks': [61, 62, 63, 65, 66, 67],
                'description': 'Presses lips together - suppression/control',
                'muscles': ['orbicularis_oris'],
                'stress_relevance': 0.78,
                'calculation_method': 'vertical_lip_compression'
            }
        }
        
        # Calibration and normalization parameters
        self.baseline_measurements = {}
        self.calibration_frames = deque(maxlen=30)  # Store 30 frames for baseline
        self.face_size_normalizer = 1.0
        self.is_calibrated = False
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.detection_confidence_history = deque(maxlen=50)
        
        # Temporal smoothing for micro-expressions
        self.au_history = {au: deque(maxlen=10) for au in self.action_unit_definitions.keys()}
        self.micro_expression_window = 0.5  # seconds
        
    def extract_action_units(self, face_landmarks, confidence_score=1.0, timestamp=None):
        """Extract and quantify facial action units from MediaPipe landmarks"""
        start_time = datetime.now()
        
        if face_landmarks is None or len(face_landmarks) != 68:
            return self._get_empty_au_result()
            
        # Convert landmarks to numpy array for processing
        landmarks = np.array(face_landmarks)
        
        # Normalize landmarks for face size and position
        normalized_landmarks = self._normalize_landmarks(landmarks)
        
        # Calibrate if needed
        if not self.is_calibrated:
            self._update_calibration(normalized_landmarks)
        
        # Extract action units with enhanced calculation methods
        action_units = {}
        reliability_scores = {}
        
        for au_code, au_info in self.action_unit_definitions.items():
            try:
                intensity = self._calculate_au_intensity_enhanced(au_code, normalized_landmarks, au_info)
                reliability = self._calculate_au_reliability(au_code, intensity, confidence_score)
                
                # Apply temporal smoothing
                smoothed_intensity = self._apply_temporal_smoothing(au_code, intensity, timestamp)
                
                action_units[au_code] = smoothed_intensity
                reliability_scores[au_code] = reliability
                
            except Exception as e:
                # Graceful degradation
                action_units[au_code] = 0.0
                reliability_scores[au_code] = 0.0
        
        # Performance tracking
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_times.append(processing_time)
        self.detection_confidence_history.append(confidence_score)
        
        return {
            'action_units': action_units,
            'reliability_scores': reliability_scores,
            'processing_time': processing_time,
            'face_confidence': confidence_score,
            'calibration_status': self.is_calibrated
        }
    
    def _normalize_landmarks(self, landmarks):
        """Advanced landmark normalization accounting for head pose and face size"""
        # Calculate face center (average of key facial points)
        face_center = np.mean(landmarks[[30, 8, 36, 45]], axis=0)  # Nose tip, chin, eye corners
        
        # Calculate face dimensions for size normalization
        # Width: distance between outer eye corners
        face_width = np.linalg.norm(landmarks[45] - landmarks[36])
        # Height: forehead to chin
        face_height = np.linalg.norm(landmarks[8] - landmarks[19])
        
        # Use average dimension for normalization
        face_size = (face_width + face_height) / 2
        self.face_size_normalizer = face_size
        
        # Normalize landmarks
        normalized = (landmarks - face_center) / face_size
        
        return normalized
    
    def _calculate_au_intensity_enhanced(self, au_code, landmarks, au_info):
        """Enhanced AU intensity calculation with method-specific algorithms"""
        method = au_info['calculation_method']
        relevant_points = [landmarks[i] for i in au_info['landmarks']]
        
        if method == 'vertical_displacement':
            # For brow raising (AU1, AU2)
            return self._calculate_vertical_displacement(relevant_points, au_code)
            
        elif method == 'vertical_displacement_and_convergence':
            # For brow lowering (AU4)
            vertical = self._calculate_vertical_displacement(relevant_points, au_code, direction='down')
            convergence = self._calculate_brow_convergence(relevant_points)
            return (vertical + convergence) / 2
            
        elif method == 'eye_area_reduction':
            # For lid tightening (AU7)
            return self._calculate_eye_area_change(relevant_points)
            
        elif method == 'lip_compression':
            # For lip tightening (AU23)
            return self._calculate_lip_area_change(relevant_points)
            
        elif method == 'corner_depression':
            # For lip corner depression (AU15)
            return self._calculate_corner_movement(relevant_points, direction='down')
            
        else:
            # Generic calculation for other AUs
            return self._calculate_generic_displacement(relevant_points)
    
    def _calculate_vertical_displacement(self, points, au_code, direction='up'):
        """Calculate vertical movement of facial points"""
        if not points or au_code not in self.baseline_measurements:
            return 0.0
            
        baseline_y = self.baseline_measurements[au_code]['mean_y']
        current_y = np.mean([p[1] for p in points])
        
        displacement = baseline_y - current_y if direction == 'up' else current_y - baseline_y
        
        # Normalize and clamp to [0, 1]
        normalized_displacement = displacement * 10  # Scale factor
        return max(0.0, min(1.0, normalized_displacement))
    
    def _calculate_brow_convergence(self, points):
        """Calculate how much eyebrows are drawn together"""
        if len(points) < 4:
            return 0.0
            
        # Distance between inner brow points
        inner_distance = np.linalg.norm(points[3] - points[0])  # Approximate inner points
        
        if 'AU4' not in self.baseline_measurements:
            return 0.0
            
        baseline_distance = self.baseline_measurements['AU4'].get('inner_distance', inner_distance)
        convergence = max(0.0, (baseline_distance - inner_distance) * 5)
        
        return min(1.0, convergence)
    
    def _calculate_eye_area_change(self, points):
        """Calculate reduction in eye opening area"""
        if len(points) < 6:
            return 0.0
            
        # Approximate eye area using height and width
        eye_height = np.mean([np.linalg.norm(points[i] - points[i+3]) for i in range(0, min(3, len(points)-3))])
        
        if 'AU7' not in self.baseline_measurements:
            return 0.0
            
        baseline_height = self.baseline_measurements['AU7'].get('eye_height', eye_height)
        area_reduction = max(0.0, (baseline_height - eye_height) * 8)
        
        return min(1.0, area_reduction)
    
    def _calculate_lip_area_change(self, points):
        """Calculate compression/tightening of lips"""
        if len(points) < 8:
            return 0.0
            
        # Calculate lip area approximation
        lip_width = np.linalg.norm(points[0] - points[6]) if len(points) > 6 else 0
        lip_height = np.mean([np.linalg.norm(points[i] - points[i+6]) for i in range(0, min(2, len(points)-6))])
        
        lip_area = lip_width * lip_height
        
        if 'AU23' not in self.baseline_measurements:
            return 0.0
            
        baseline_area = self.baseline_measurements['AU23'].get('lip_area', lip_area)
        compression = max(0.0, (baseline_area - lip_area) * 6)
        
        return min(1.0, compression)
    
    def _calculate_corner_movement(self, points, direction='down'):
        """Calculate movement of lip corners"""
        if len(points) < 2:
            return 0.0
            
        # Average vertical position of corner points
        corner_y = np.mean([p[1] for p in points])
        
        if 'AU15' not in self.baseline_measurements:
            return 0.0
            
        baseline_y = self.baseline_measurements['AU15']['corner_y']
        movement = corner_y - baseline_y if direction == 'down' else baseline_y - corner_y
        
        normalized_movement = movement * 8  # Scale factor
        return max(0.0, min(1.0, normalized_movement))
    
    def _calculate_generic_displacement(self, points):
        """Generic displacement calculation for less critical AUs"""
        if not points:
            return 0.0
            
        # Calculate variance in point positions as activity indicator
        point_array = np.array(points)
        variance = np.var(point_array)
        
        # Normalize to reasonable range
        normalized_variance = min(1.0, variance * 50)
        return max(0.0, normalized_variance)
    
    def _update_calibration(self, landmarks):
        """Update baseline measurements for AU calibration"""
        self.calibration_frames.append(landmarks.copy())
        
        if len(self.calibration_frames) >= 10:  # Need minimum frames for stable baseline
            # Calculate baseline measurements for each AU
            stacked_frames = np.array(list(self.calibration_frames))
            
            for au_code, au_info in self.action_unit_definitions.items():
                relevant_indices = au_info['landmarks']
                relevant_points = stacked_frames[:, relevant_indices, :]
                
                # Calculate baseline statistics
                self.baseline_measurements[au_code] = {
                    'mean_x': np.mean(relevant_points[:, :, 0]),
                    'mean_y': np.mean(relevant_points[:, :, 1]),
                    'std_x': np.std(relevant_points[:, :, 0]),
                    'std_y': np.std(relevant_points[:, :, 1])
                }
                
                # AU-specific baseline measurements
                if au_code == 'AU4':  # Brow convergence baseline
                    inner_distances = [np.linalg.norm(frame[19] - frame[22]) for frame in stacked_frames]
                    self.baseline_measurements[au_code]['inner_distance'] = np.mean(inner_distances)
                    
                elif au_code == 'AU7':  # Eye area baseline
                    eye_heights = [np.mean([np.linalg.norm(frame[37] - frame[41]), 
                                           np.linalg.norm(frame[43] - frame[47])]) for frame in stacked_frames]
                    self.baseline_measurements[au_code]['eye_height'] = np.mean(eye_heights)
                    
                elif au_code == 'AU23':  # Lip area baseline
                    lip_widths = [np.linalg.norm(frame[48] - frame[54]) for frame in stacked_frames]
                    lip_heights = [np.linalg.norm(frame[51] - frame[57]) for frame in stacked_frames]
                    lip_areas = [w * h for w, h in zip(lip_widths, lip_heights)]
                    self.baseline_measurements[au_code]['lip_area'] = np.mean(lip_areas)
                    
                elif au_code == 'AU15':  # Corner position baseline
                    corner_ys = [np.mean([frame[48][1], frame[54][1]]) for frame in stacked_frames]
                    self.baseline_measurements[au_code]['corner_y'] = np.mean(corner_ys)
            
            self.is_calibrated = True
    
    def _calculate_au_reliability(self, au_code, intensity, face_confidence):
        """Calculate reliability score for AU detection"""
        au_info = self.action_unit_definitions[au_code]
        base_reliability = au_info['stress_relevance']
        
        # Factor in face detection confidence
        confidence_factor = face_confidence
        
        # Factor in calibration status
        calibration_factor = 1.0 if self.is_calibrated else 0.7
        
        # Factor in intensity (very low intensities are less reliable)
        intensity_factor = min(1.0, intensity * 2 + 0.5)
        
        combined_reliability = base_reliability * confidence_factor * calibration_factor * intensity_factor
        
        return min(1.0, combined_reliability)
    
    def _apply_temporal_smoothing(self, au_code, intensity, timestamp=None):
        """Apply temporal smoothing to reduce noise in AU detection"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Store current measurement
        self.au_history[au_code].append({
            'intensity': intensity,
            'timestamp': timestamp
        })
        
        # Get recent measurements within time window
        recent_measurements = []
        cutoff_time = timestamp - timedelta(seconds=self.micro_expression_window)
        
        for measurement in self.au_history[au_code]:
            if measurement['timestamp'] > cutoff_time:
                recent_measurements.append(measurement['intensity'])
        
        if len(recent_measurements) < 2:
            return intensity
        
        # Apply weighted smoothing (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(recent_measurements))
        smoothed_intensity = np.average(recent_measurements, weights=weights)
        
        return smoothed_intensity
    
    def _get_empty_au_result(self):
        """Return empty result when face detection fails"""
        empty_aus = {au: 0.0 for au in self.action_unit_definitions.keys()}
        empty_reliability = {au: 0.0 for au in self.action_unit_definitions.keys()}
        
        return {
            'action_units': empty_aus,
            'reliability_scores': empty_reliability,
            'processing_time': 0.0,
            'face_confidence': 0.0,
            'calibration_status': False
        }
    
    def get_stress_priority_aus(self):
        """Get Action Units most relevant for stress detection"""
        stress_priority = {}
        for au_code, au_info in self.action_unit_definitions.items():
            if au_info['stress_relevance'] >= 0.8:
                stress_priority[au_code] = {
                    'name': au_info['name'],
                    'description': au_info['description'],
                    'relevance': au_info['stress_relevance']
                }
        return stress_priority
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring"""
        if not self.processing_times:
            return {'status': 'no_data'}
            
        return {
            'avg_processing_time': np.mean(list(self.processing_times)),
            'max_processing_time': np.max(list(self.processing_times)),
            'avg_face_confidence': np.mean(list(self.detection_confidence_history)) if self.detection_confidence_history else 0.0,
            'calibration_frames': len(self.calibration_frames),
            'is_calibrated': self.is_calibrated,
            'measurements_count': len(self.processing_times)
        }
    
    def reset_calibration(self):
        """Reset calibration for new user or environment"""
        self.calibration_frames.clear()
        self.baseline_measurements.clear()
        self.is_calibrated = False
        
        # Clear AU history
        for au_code in self.au_history:
            self.au_history[au_code].clear()