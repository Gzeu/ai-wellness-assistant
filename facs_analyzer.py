#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 FACS Analyzer - Facial Action Coding System
AI Wellness Assistant - Stress Detection Module

Implementează sistemul FACS pentru detectarea și cuantificarea
unităților de acțiune facială pentru analiza stresului.

Author: Pricop George
Date: 21 Septembrie 2025
Version: 1.0.0
Linear: GPZ-40
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ActionUnit:
    """Reprezentare pentru o unitate de acțiune facială FACS"""
    id: int
    name: str
    description: str
    landmarks: List[int]
    weight: float
    stress_indicator: bool = False

@dataclass
class FACSResult:
    """Rezultatul analizei FACS"""
    action_units: Dict[int, float]  # AU_ID -> intensity (0-1)
    confidence: float
    timestamp: float
    stress_indicators: Dict[str, float]
    micro_expressions: List[str]

class FACSAnalyzer:
    """🔬 FACS Analyzer - Facial Action Coding System Implementation
    
    Implementează detectarea și cuantificarea unităților de acțiune facială
    conform sistemului dezvoltat de Paul Ekman și Wallace Friesen.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Inițializare FACS Analyzer
        
        Args:
            confidence_threshold: Pragul minim de încredere pentru detecții
        """
        self.confidence_threshold = confidence_threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Inițializare action units
        self.action_units = self._initialize_action_units()
        
        # Istoric pentru temporal analysis
        self.history = defaultdict(list)
        self.max_history = 30  # 30 frames pentru smoothing
        
        logger.info("🔬 FACS Analyzer initialized with %d action units", len(self.action_units))
    
    def _initialize_action_units(self) -> Dict[int, ActionUnit]:
        """Inițializare action units cu landmarks MediaPipe"""
        
        action_units = {
            # ⚡ Priority Action Units pentru Stress Detection
            1: ActionUnit(1, "Inner Brow Raiser", "Ridică sprâncenele interioare", 
                         [55, 65], 0.9, stress_indicator=True),
                         
            4: ActionUnit(4, "Brow Lowerer", "Coborâre sprâncene (tensiune)", 
                         [55, 65, 70], 0.85, stress_indicator=True),
                         
            7: ActionUnit(7, "Lid Tightener", "Strângere pleoape (anxietate)", 
                         [33, 7, 163, 144], 0.8, stress_indicator=True),
                         
            15: ActionUnit(15, "Lip Corner Depressor", "Coborâre colțuri gură", 
                          [61, 291, 17, 18], 0.75, stress_indicator=True),
                          
            23: ActionUnit(23, "Lip Tightener", "Strângere buze (tensiune)", 
                          [78, 95, 88, 178], 0.8, stress_indicator=True),
            
            # 🔍 Secondary Action Units
            2: ActionUnit(2, "Outer Brow Raiser", "Ridică sprâncenele exterioare", 
                         [46, 53, 276, 283], 0.7),
                         
            5: ActionUnit(5, "Upper Lid Raiser", "Ridică pleoapele superioare", 
                         [33, 7, 163, 144], 0.6),
                         
            9: ActionUnit(9, "Nose Wrinkler", "Încrețirea nasului", 
                         [115, 131, 134, 102], 0.7),
                         
            10: ActionUnit(10, "Upper Lip Raiser", "Ridică buza superioară", 
                          [72, 6, 302], 0.65),
                          
            17: ActionUnit(17, "Chin Raiser", "Ridică bărbia", 
                          [175, 199, 208, 428], 0.6),
                          
            24: ActionUnit(24, "Lip Pressor", "Presiune pe buze", 
                          [78, 95, 88, 178], 0.75, stress_indicator=True),
                          
            # 📊 Additional Detection Units
            12: ActionUnit(12, "Lip Corner Puller", "Tragere colțuri gură (zâmbet)", 
                          [61, 291, 39, 269], 0.8),
                          
            25: ActionUnit(25, "Lips Part", "Deschidere ușoară a buzelor", 
                          [13, 14, 15, 16], 0.5),
                          
            26: ActionUnit(26, "Jaw Drop", "Deschidere gură (surpriză)", 
                          [13, 14, 15, 16, 17], 0.7)
        }
        
        return action_units
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detectare facial landmarks folosind MediaPipe
        
        Args:
            frame: Frame-ul video de analizat
            
        Returns:
            Array cu landmarks (468 puncte) sau None dacă nu se detectează față
        """
        try:
            # Conversie la RGB pentru MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesare cu MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Extrage landmarks pentru prima față detectată
                landmarks = results.multi_face_landmarks[0]
                
                # Conversie la coordonate normalizate
                h, w = frame.shape[:2]
                landmark_array = np.array([
                    [lm.x * w, lm.y * h, lm.z] for lm in landmarks.landmark
                ])
                
                return landmark_array
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Eroare detectare landmarks: {e}")
            return None
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculează distanța euclidienă între două puncte"""
        return np.linalg.norm(p1[:2] - p2[:2])  # Folosește doar x,y
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculează unghiul format de trei puncte"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_action_unit_intensity(self, au: ActionUnit, landmarks: np.ndarray) -> float:
        """Calculează intensitatea unei action unit specifice
        
        Args:
            au: Action unit de calculat
            landmarks: Array-ul cu facial landmarks
            
        Returns:
            Intensitatea AU pe scala 0-1
        """
        try:
            if len(au.landmarks) < 2:
                return 0.0
            
            # Extrage punctele relevante
            points = [landmarks[idx] for idx in au.landmarks if idx < len(landmarks)]
            
            if len(points) < 2:
                return 0.0
            
            # Calculează intensitatea bazată pe tipul AU
            if au.id == 1:  # Inner Brow Raiser
                # Distanța între sprâncenele interioare și centrul ochilor
                left_brow = points[0]
                right_brow = points[1] if len(points) > 1 else points[0]
                eye_center = landmarks[168] if len(landmarks) > 168 else points[0]
                
                distance = self.calculate_distance(left_brow, eye_center)
                intensity = min(distance / 20.0, 1.0)  # Normalizare
                
            elif au.id == 4:  # Brow Lowerer
                # Unghiul sprâncenelor față de orizontală
                if len(points) >= 3:
                    angle = self.calculate_angle(points[0], points[1], points[2])
                    intensity = max(0, min((180 - angle) / 30.0, 1.0))
                else:
                    intensity = 0.0
                    
            elif au.id == 7:  # Lid Tightener
                # Distanța între pleoapele superioare și inferioare
                if len(points) >= 4:
                    upper_lid = (points[0] + points[1]) / 2
                    lower_lid = (points[2] + points[3]) / 2
                    distance = self.calculate_distance(upper_lid, lower_lid)
                    intensity = max(0, 1.0 - distance / 15.0)  # Inversă
                else:
                    intensity = 0.0
                    
            elif au.id == 15:  # Lip Corner Depressor
                # Poziția colțurilor gurii față de linia neutră
                if len(points) >= 4:
                    left_corner = points[0]
                    right_corner = points[1]
                    lip_center = (points[2] + points[3]) / 2
                    
                    # Calculează cât de mult sunt coborâte colțurile
                    left_depression = max(0, lip_center[1] - left_corner[1])
                    right_depression = max(0, lip_center[1] - right_corner[1])
                    
                    intensity = min((left_depression + right_depression) / 20.0, 1.0)
                else:
                    intensity = 0.0
                    
            elif au.id == 23 or au.id == 24:  # Lip Tightener/Pressor
                # Grosimea buzelor și tensiunea
                if len(points) >= 4:
                    lip_thickness = self.calculate_distance(points[0], points[2])
                    normal_thickness = 8.0  # Valoare de referință
                    intensity = max(0, (normal_thickness - lip_thickness) / normal_thickness)
                else:
                    intensity = 0.0
                    
            else:
                # Algoritm generic bazat pe variația pozițiilor
                if len(points) >= 2:
                    # Calculează variația față de poziția neutră estimată
                    center = np.mean(points, axis=0)
                    variation = np.std([self.calculate_distance(p, center) for p in points])
                    intensity = min(variation / 10.0, 1.0)
                else:
                    intensity = 0.0
            
            # Aplică smoothing temporal
            intensity = self._apply_temporal_smoothing(au.id, intensity)
            
            return max(0.0, min(1.0, intensity))
            
        except Exception as e:
            logger.error(f"❌ Eroare calculare AU{au.id}: {e}")
            return 0.0
    
    def _apply_temporal_smoothing(self, au_id: int, intensity: float) -> float:
        """Aplică smoothing temporal pentru reducerea zgomotului"""
        try:
            # Adaugă în istoric
            self.history[au_id].append(intensity)
            
            # Limitează mărimea istoricului
            if len(self.history[au_id]) > self.max_history:
                self.history[au_id].pop(0)
            
            # Calculează media ponderată (prioritate pentru valorile recente)
            if len(self.history[au_id]) < 3:
                return intensity
            
            weights = np.exp(np.linspace(-2, 0, len(self.history[au_id])))
            weights /= weights.sum()
            
            smoothed = np.average(self.history[au_id], weights=weights)
            
            return float(smoothed)
            
        except Exception:
            return intensity
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[FACSResult]:
        """🔍 Analizează un frame pentru action units
        
        Args:
            frame: Frame-ul video de analizat
            
        Returns:
            FACSResult cu intensitățile AU detectate sau None
        """
        try:
            # Detectare landmarks
            landmarks = self.detect_landmarks(frame)
            
            if landmarks is None:
                logger.warning("⚠️ Nu s-au detectat landmarks faciale")
                return None
            
            # Calculare intensități pentru toate action units
            au_intensities = {}
            stress_indicators = {}
            micro_expressions = []
            
            total_confidence = 0.0
            
            for au_id, au in self.action_units.items():
                intensity = self.calculate_action_unit_intensity(au, landmarks)
                au_intensities[au_id] = intensity
                
                # Verifică dacă este indicator de stress
                if au.stress_indicator and intensity > 0.3:
                    stress_indicators[au.name] = intensity
                    
                    # Detectare micro-expresii (intensitate mare și rapidă)
                    if intensity > 0.7:
                        micro_expressions.append(f"AU{au_id}: {au.name}")
                
                # Calculare confidence cumulativă
                total_confidence += intensity * au.weight
            
            # Normalizare confidence
            max_possible_confidence = sum(au.weight for au in self.action_units.values())
            confidence = min(total_confidence / max_possible_confidence, 1.0)
            
            return FACSResult(
                action_units=au_intensities,
                confidence=confidence,
                timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
                stress_indicators=stress_indicators,
                micro_expressions=micro_expressions
            )
            
        except Exception as e:
            logger.error(f"❌ Eroare analiză frame FACS: {e}")
            return None
    
    def get_stress_level(self, facs_result: FACSResult) -> Tuple[float, str, Dict]:
        """📊 Calculează nivelul de stress din rezultatul FACS
        
        Args:
            facs_result: Rezultatul analizei FACS
            
        Returns:
            Tuple cu (stress_score, stress_level, details)
        """
        if not facs_result or not facs_result.stress_indicators:
            return 0.0, "Relaxat", {}
        
        # Calculare stress score ponderat
        stress_score = 0.0
        pattern_details = {}
        
        # Identificare pattern-uri specifice de stress
        tension_aus = [4, 7, 23, 24]  # Tension pattern
        anxiety_aus = [1, 2, 5]       # Anxiety pattern
        suppression_aus = [15, 17, 24] # Suppression pattern
        
        # Calculare pentru fiecare pattern
        patterns = {
            'tension': (tension_aus, 0.8),
            'anxiety': (anxiety_aus, 0.9),
            'suppression': (suppression_aus, 0.7)
        }
        
        for pattern_name, (aus, weight) in patterns.items():
            pattern_intensity = np.mean([
                facs_result.action_units.get(au_id, 0.0) for au_id in aus
            ])
            
            if pattern_intensity > 0.2:
                pattern_score = pattern_intensity * weight
                stress_score += pattern_score
                pattern_details[pattern_name] = {
                    'intensity': pattern_intensity,
                    'score': pattern_score,
                    'active_units': [au_id for au_id in aus 
                                   if facs_result.action_units.get(au_id, 0.0) > 0.2]
                }
        
        # Normalizare stress score (0-1)
        stress_score = min(stress_score, 1.0)
        
        # Clasificare nivel stress
        if stress_score < 0.2:
            stress_level = "Relaxat 😌"
        elif stress_score < 0.4:
            stress_level = "Ușor stres 😐"
        elif stress_score < 0.6:
            stress_level = "Stres moderat 😟"
        elif stress_score < 0.8:
            stress_level = "Stres ridicat 😰"
        else:
            stress_level = "Stres extrem 😱"
        
        return stress_score, stress_level, pattern_details
    
    def analyze_video_stream(self, video_source: int = 0) -> None:
        """🎥 Analiză în timp real dintr-un stream video
        
        Args:
            video_source: Indexul camerei (0 = camera implicită)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error("❌ Nu se poate deschide camera")
            return
        
        logger.info("🎥 Începe analiza video stream FACS...")
        logger.info("Apasă 'q' pentru oprire, 's' pentru screenshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analiză FACS
                facs_result = self.analyze_frame(frame)
                
                if facs_result:
                    # Calculare stress level
                    stress_score, stress_level, details = self.get_stress_level(facs_result)
                    
                    # Overlay informații pe frame
                    frame = self._draw_analysis_overlay(frame, facs_result, stress_score, stress_level)
                    
                    # Log rezultate
                    if stress_score > 0.3:  # Doar dacă e detectat stress
                        logger.info(f"📊 Stress: {stress_score:.2f} | {stress_level} | Confidence: {facs_result.confidence:.2f}")
                        
                        if facs_result.micro_expressions:
                            logger.info(f"⚡ Micro-expresii: {', '.join(facs_result.micro_expressions)}")
                
                # Afișare frame
                cv2.imshow('FACS Analyzer - AI Wellness Assistant', frame)
                
                # Controale keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'facs_screenshot_{int(cv2.getTickCount())}.jpg', frame)
                    logger.info("📸 Screenshot salvat!")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("🏁 Analiza FACS încheiată")
    
    def _draw_analysis_overlay(self, frame: np.ndarray, facs_result: FACSResult, 
                              stress_score: float, stress_level: str) -> np.ndarray:
        """Desenează overlay cu informații de analiză pe frame"""
        try:
            overlay_frame = frame.copy()
            
            # Background pentru text
            cv2.rectangle(overlay_frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(overlay_frame, (10, 10), (400, 120), (255, 255, 255), 2)
            
            # Text principal
            cv2.putText(overlay_frame, f"Stress Level: {stress_level}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(overlay_frame, f"Score: {stress_score:.2f} | Confidence: {facs_result.confidence:.2f}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Action units active
            active_aus = [f"AU{au_id}:{intensity:.2f}" 
                         for au_id, intensity in facs_result.action_units.items() 
                         if intensity > 0.2]
            
            if active_aus:
                au_text = " | ".join(active_aus[:4])  # Primele 4
                cv2.putText(overlay_frame, au_text, (20, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Indicator de stress colorat
            color = (0, 255, 0)  # Verde pentru relaxat
            if stress_score > 0.4:
                color = (0, 255, 255)  # Galben pentru moderat
            if stress_score > 0.7:
                color = (0, 0, 255)    # Roșu pentru ridicat
            
            cv2.circle(overlay_frame, (380, 30), 15, color, -1)
            
            return overlay_frame
            
        except Exception as e:
            logger.error(f"❌ Eroare overlay: {e}")
            return frame
    
    def export_analysis_results(self, results: List[FACSResult], filename: str = "facs_analysis.json"):
        """📁 Exportă rezultatele analizei în format JSON"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                "analysis_session": {
                    "timestamp": datetime.now().isoformat(),
                    "total_frames": len(results),
                    "analyzer_version": "1.0.0",
                    "action_units_detected": len(self.action_units)
                },
                "results": [{
                    "timestamp": result.timestamp,
                    "action_units": result.action_units,
                    "confidence": result.confidence,
                    "stress_indicators": result.stress_indicators,
                    "micro_expressions": result.micro_expressions
                } for result in results if result]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Rezultate exportate în {filename}")
            
        except Exception as e:
            logger.error(f"❌ Eroare export: {e}")


# 🧪 Demo și Testing
if __name__ == "__main__":
    print("🧠 FACS Analyzer - AI Wellness Assistant")
    print("🔬 Facial Action Coding System Implementation")
    print("📅 21 Septembrie 2025 - GPZ-40")
    print("-" * 50)
    
    # Inițializare analyzer
    analyzer = FACSAnalyzer(confidence_threshold=0.7)
    
    print(f"✅ FACS Analyzer inițializat cu {len(analyzer.action_units)} action units")
    print(f"🎯 Action units pentru stress: {sum(1 for au in analyzer.action_units.values() if au.stress_indicator)}")
    
    # Demonstrație
    print("\n🎥 Pentru a începe analiza video, rulează:")
    print("python facs_analyzer.py")
    print("\n📋 Pentru integrare cu StressDetector din GPZ-41:")
    print("from facs_analyzer import FACSAnalyzer")
    print("analyzer = FACSAnalyzer()")
    
    # Test pe cameră dacă este disponibilă
    try:
        import argparse
        parser = argparse.ArgumentParser(description='FACS Analyzer Demo')
        parser.add_argument('--demo', action='store_true', help='Rulează demo live')
        parser.add_argument('--test', action='store_true', help='Rulează teste unitare')
        
        args = parser.parse_args()
        
        if args.demo:
            print("\n🎥 Începe demo live cu camera...")
            analyzer.analyze_video_stream(0)
        elif args.test:
            print("\n🧪 Rulează teste unitare...")
            # TODO: Implementare teste
            pass
        else:
            print("\n💡 Utilizare: python facs_analyzer.py [--demo|--test]")
    
    except ImportError:
        print("\n💡 Pentru demo live: pip install opencv-python mediapipe")
