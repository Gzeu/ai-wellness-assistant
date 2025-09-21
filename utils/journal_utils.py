"""
Emotional Journal Manager
Privacy-first journaling system for mood and stress tracking with insights
Author: Pricop George - AI Wellness Assistant
"""

import json
import sqlite3
import os
from datetime import datetime, timedelta, date
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Any

class JournalManager:
    def __init__(self, db_path='data/wellness_journal.db', backup_path='data/journal_backup.json'):
        self.db_path = db_path
        self.backup_path = backup_path
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Mood and stress level mappings
        self.mood_levels = {
            1: {'name': 'Foarte trist', 'color': '#F44336', 'emoji': '😢'},
            2: {'name': 'Trist', 'color': '#FF5722', 'emoji': '🙁'},
            3: {'name': 'Nemulțumit', 'color': '#FF9800', 'emoji': '😐'},
            4: {'name': 'OK', 'color': '#FFC107', 'emoji': '🙂'},
            5: {'name': 'Neutru', 'color': '#9E9E9E', 'emoji': '😑'},
            6: {'name': 'Bine', 'color': '#4CAF50', 'emoji': '😊'},
            7: {'name': 'Mulțumit', 'color': '#8BC34A', 'emoji': '😄'},
            8: {'name': 'Fericit', 'color': '#CDDC39', 'emoji': '😁'},
            9: {'name': 'Foarte fericit', 'color': '#FFEB3B', 'emoji': '😆'},
            10: {'name': 'Euforic', 'color': '#FFF176', 'emoji': '🤩'}
        }
        
        self.stress_levels = {
            1: {'name': 'Foarte relaxat', 'color': '#4CAF50', 'description': 'Calm complet'},
            2: {'name': 'Relaxat', 'color': '#8BC34A', 'description': 'Foarte calm'},
            3: {'name': 'Ușor tensionat', 'color': '#CDDC39', 'description': 'Puțină tensiune'},
            4: {'name': 'Moderat', 'color': '#FFEB3B', 'description': 'Stress ușor'},
            5: {'name': 'Tensionat', 'color': '#FFC107', 'description': 'Stress moderat'},
            6: {'name': 'Stresant', 'color': '#FF9800', 'description': 'Stress ridicat'},
            7: {'name': 'Foarte stresant', 'color': '#FF5722', 'description': 'Stress foarte ridicat'},
            8: {'name': 'Anxios', 'color': '#F44336', 'description': 'Anxietate puternică'},
            9: {'name': 'Foarte anxios', 'color': '#D32F2F', 'description': 'Anxietate extremă'},
            10: {'name': 'Panică', 'color': '#B71C1C', 'description': 'Stare de panică'}
        }
        
    def _init_database(self):
        """Initialize SQLite database with comprehensive schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    mood INTEGER NOT NULL CHECK(mood >= 1 AND mood <= 10),
                    stress_level INTEGER NOT NULL CHECK(stress_level >= 1 AND stress_level <= 10),
                    notes TEXT,
                    highlights TEXT,
                    detected_stress_score REAL,
                    detected_patterns TEXT,  -- JSON string
                    techniques_used TEXT,    -- JSON string
                    technique_effectiveness TEXT,  -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for efficient queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON journal_entries(date)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON journal_entries(created_at)')
            
    def add_entry(self, mood: int, stress_level: int, notes: str = "", highlights: str = "", 
                  detected_stress_score: float = None, detected_patterns: List = None,
                  techniques_used: List = None, technique_effectiveness: Dict = None) -> Dict:
        """Add new journal entry with comprehensive validation"""
        
        # Validate inputs
        if not (1 <= mood <= 10):
            raise ValueError("Mood must be between 1 and 10")
        if not (1 <= stress_level <= 10):
            raise ValueError("Stress level must be between 1 and 10")
        
        entry_date = datetime.now().date().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO journal_entries 
                (date, mood, stress_level, notes, highlights, detected_stress_score, 
                 detected_patterns, techniques_used, technique_effectiveness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry_date, mood, stress_level, notes.strip(), highlights.strip(),
                detected_stress_score,
                json.dumps(detected_patterns) if detected_patterns else None,
                json.dumps(techniques_used) if techniques_used else None,
                json.dumps(technique_effectiveness) if technique_effectiveness else None
            ))
            
            entry_id = cursor.lastrowid
        
        # Auto-backup
        self._create_backup()
        
        return {
            'id': entry_id,
            'success': True,
            'message': 'Entry added successfully',
            'date': entry_date,
            'mood': mood,
            'stress_level': stress_level
        }
    
    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        """Get most recent journal entries with full details"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM journal_entries 
                ORDER BY date DESC, created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            entries = []
            for row in cursor.fetchall():
                entry = dict(row)
                
                # Parse JSON fields
                if entry['detected_patterns']:
                    entry['detected_patterns'] = json.loads(entry['detected_patterns'])
                if entry['techniques_used']:
                    entry['techniques_used'] = json.loads(entry['techniques_used'])
                if entry['technique_effectiveness']:
                    entry['technique_effectiveness'] = json.loads(entry['technique_effectiveness'])
                    
                # Add interpreted levels
                entry['mood_info'] = self.mood_levels.get(entry['mood'], {})
                entry['stress_info'] = self.stress_levels.get(entry['stress_level'], {})
                
                entries.append(entry)
        
        return entries
    
    def generate_insights(self, days_back: int = 30) -> Dict:
        """Generate comprehensive insights from journal data"""
        entries = self.get_recent_entries(days_back * 2)  # Get more entries for analysis
        
        if not entries:
            return {
                'status': 'no_data',
                'message': 'Nu există date suficiente pentru insights'
            }
        
        # Filter by date range
        cutoff_date = (datetime.now() - timedelta(days=days_back)).date().isoformat()
        entries = [e for e in entries if e['date'] >= cutoff_date]
        
        insights = {
            'period_summary': {
                'total_entries': len(entries),
                'days_tracked': len(set(entry['date'] for entry in entries)),
                'tracking_consistency': len(set(entry['date'] for entry in entries)) / days_back
            },
            'mood_analysis': self._analyze_mood_trends(entries),
            'stress_analysis': self._analyze_stress_patterns(entries),
            'progress_indicators': self._identify_progress_indicators(entries),
            'recommendations': self._generate_recommendations(entries)
        }
        
        return insights
    
    def _analyze_mood_trends(self, entries: List[Dict]) -> Dict:
        """Analyze mood trends with detailed statistics"""
        moods = [entry['mood'] for entry in entries]
        
        if len(moods) < 2:
            return {'status': 'insufficient_data'}
        
        # Trend calculation
        x = np.arange(len(moods))
        trend_slope = np.polyfit(x, moods, 1)[0]
        
        return {
            'average_mood': round(np.mean(moods), 2),
            'mood_range': {'min': min(moods), 'max': max(moods)},
            'trend_slope': round(trend_slope, 3),
            'trend_direction': 'improving' if trend_slope > 0.1 else 'declining' if trend_slope < -0.1 else 'stable',
            'mood_variability': round(np.std(moods), 2),
            'best_mood': max(moods),
            'worst_mood': min(moods)
        }
    
    def _analyze_stress_patterns(self, entries: List[Dict]) -> Dict:
        """Analyze stress patterns and triggers"""
        stress_levels = [entry['stress_level'] for entry in entries]
        
        if len(stress_levels) < 2:
            return {'status': 'insufficient_data'}
        
        return {
            'average_stress': round(np.mean(stress_levels), 2),
            'stress_range': {'min': min(stress_levels), 'max': max(stress_levels)},
            'high_stress_days': sum(1 for s in stress_levels if s >= 7),
            'low_stress_days': sum(1 for s in stress_levels if s <= 3),
            'stress_variability': round(np.std(stress_levels), 2)
        }
    
    def _identify_progress_indicators(self, entries: List[Dict]) -> List[Dict]:
        """Identify positive progress and achievements"""
        achievements = []
        
        if len(entries) >= 7:
            achievements.append({
                'type': 'consistency',
                'title': '7 zile de tracking! 🏆',
                'description': 'Consistență excelentă în jurnal'
            })
        
        # Mood improvements
        if len(entries) >= 10:
            recent_moods = [e['mood'] for e in entries[:5]]
            older_moods = [e['mood'] for e in entries[5:10]]
            if np.mean(recent_moods) > np.mean(older_moods) + 0.5:
                achievements.append({
                    'type': 'improvement',
                    'title': 'Îmbunătățire mood! 😊',
                    'description': 'Starea de spirit s-a îmbunătățit recent'
                })
        
        return achievements
    
    def _generate_recommendations(self, entries: List[Dict]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if not entries:
            return ['Începe să completezi jurnalul pentru insights personalizate']
        
        # Stress level recommendations
        avg_stress = np.mean([e['stress_level'] for e in entries])
        if avg_stress > 6:
            recommendations.append('Nivelul de stress este ridicat - consideră tehnici de relaxare zilnice')
        
        # Consistency recommendations
        unique_dates = set(entry['date'] for entry in entries)
        if len(unique_dates) < 10 and len(entries) > 5:
            recommendations.append('Încearcă să completezi jurnalul mai constant pentru insights mai bune')
        
        return recommendations
    
    def _create_backup(self):
        """Create JSON backup of recent data"""
        try:
            recent_entries = self.get_recent_entries(50)
            backup_data = {
                'backup_date': datetime.now().isoformat(),
                'entries_count': len(recent_entries),
                'entries': recent_entries
            }
            
            with open(self.backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Backup failed: {e}")