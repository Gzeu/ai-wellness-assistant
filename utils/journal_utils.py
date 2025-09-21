"""
Emotional Journal Management Utilities
Handles journal entries, data persistence, and insights generation
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

class JournalManager:
    def __init__(self, data_file='static/data/journal_entries.json'):
        self.data_file = data_file
        self.entries = []
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # Load existing entries
        self.load_entries()
        
    def add_entry(self, mood: int, stress_level: int, notes: str = '', highlights: str = '') -> Dict:
        """Add new journal entry"""
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'mood': mood,  # 1-10 scale
            'stress_level': stress_level,  # 1-10 scale
            'notes': notes,
            'highlights': highlights,
            'detected_stress': None,  # Will be filled by AI analysis
            'recommended_techniques': []
        }
        
        self.entries.append(entry)
        self.save_entries()
        
        return entry
    
    def get_recent_entries(self, days: int = 30) -> List[Dict]:
        """Get recent journal entries"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent = []
        for entry in self.entries:
            entry_date = datetime.fromisoformat(entry['timestamp'])
            if entry_date > cutoff_date:
                recent.append(entry)
                
        return sorted(recent, key=lambda x: x['timestamp'], reverse=True)
    
    def generate_insights(self, days: int = 7) -> Dict:
        """Generate wellness insights from journal data"""
        recent_entries = self.get_recent_entries(days)
        
        if not recent_entries:
            return {'message': 'No data available. Start journaling to see insights!'}
            
        # Calculate averages
        moods = [entry['mood'] for entry in recent_entries]
        stress_levels = [entry['stress_level'] for entry in recent_entries]
        
        avg_mood = np.mean(moods)
        avg_stress = np.mean(stress_levels)
        
        # Trend analysis
        mood_trend = 'stable'
        if len(moods) > 3:
            recent_avg = np.mean(moods[-3:])
            earlier_avg = np.mean(moods[:-3])
            
            if recent_avg > earlier_avg + 0.5:
                mood_trend = 'improving'
            elif recent_avg < earlier_avg - 0.5:
                mood_trend = 'declining'
                
        return {
            'period_days': days,
            'total_entries': len(recent_entries),
            'average_mood': round(avg_mood, 1),
            'average_stress': round(avg_stress, 1),
            'mood_trend': mood_trend,
            'consistency': len(recent_entries) / days
        }
    
    def load_entries(self):
        """Load journal entries from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.entries = json.load(f)
        except Exception as e:
            print(f"Error loading journal entries: {e}")
            self.entries = []
            
    def save_entries(self):
        """Save journal entries to file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving journal entries: {e}")