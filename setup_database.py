#!/usr/bin/env python3
"""
Database Setup Script for AI Wellness Assistant
Initialize SQLite database with user authentication and wellness data tables

Author: George Pricop
Version: 2.0.0-AUTH
Date: September 2025
"""

import sqlite3
import os
import json
import bcrypt
import secrets
from datetime import datetime, timedelta

# Database configuration
DATABASE_PATH = "data/wellness_users.db"
DATA_DIRECTORY = "data"

def create_database_tables():
    """Create all necessary database tables"""
    print("Creating database tables...")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute('PRAGMA foreign_keys = ON')
        
        # Users table with comprehensive profile data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                profile_data TEXT DEFAULT '{}',
                preferences TEXT DEFAULT '{}',
                email_verified BOOLEAN DEFAULT FALSE,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP
            )
        ''')
        
        # User wellness sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration INTEGER DEFAULT 0,
                avg_stress_score REAL DEFAULT 0.0,
                max_stress_score REAL DEFAULT 0.0,
                min_stress_score REAL DEFAULT 0.0,
                stress_episodes INTEGER DEFAULT 0,
                face_detection_rate REAL DEFAULT 0.0,
                notes TEXT DEFAULT '',
                session_rating INTEGER,  -- User rating 1-5
                tags TEXT DEFAULT '[]',  -- JSON array of tags
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Real-time stress history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stress_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                stress_score REAL NOT NULL,
                raw_stress_score REAL,
                stress_level TEXT NOT NULL,
                face_detected BOOLEAN DEFAULT FALSE,
                confidence REAL DEFAULT 0.0,
                user_calibrated BOOLEAN DEFAULT FALSE,
                environmental_factors TEXT DEFAULT '{}',  -- JSON: lighting, noise, etc.
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES user_sessions (id) ON DELETE SET NULL
            )
        ''')
        
        # User-specific recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_recommendations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                recommendation_text TEXT NOT NULL,
                category TEXT NOT NULL,  -- breathing, exercise, mindfulness, etc.
                priority INTEGER DEFAULT 1,  -- 1-5, 1 = highest
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_at TIMESTAMP,
                used BOOLEAN DEFAULT FALSE,
                effectiveness_rating INTEGER,  -- 1-5 user rating
                feedback TEXT DEFAULT '',
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # User stress baselines and calibration data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_calibration (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                baseline_stress REAL NOT NULL,
                calibration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                calibration_readings TEXT NOT NULL,  -- JSON array of readings
                is_active BOOLEAN DEFAULT TRUE,
                notes TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # User goals and achievements
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_goals (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                goal_type TEXT NOT NULL,  -- daily_stress_avg, weekly_sessions, etc.
                goal_value REAL NOT NULL,
                current_value REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                target_date DATE,
                achieved BOOLEAN DEFAULT FALSE,
                achieved_at TIMESTAMP,
                description TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # System notifications for users
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_notifications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT DEFAULT 'info',  -- info, warning, success, error
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                read_at TIMESTAMP,
                action_url TEXT,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Create indexes for better performance
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)',
            'CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)',
            'CREATE INDEX IF NOT EXISTS idx_sessions_user_date ON user_sessions(user_id, created_at)',
            'CREATE INDEX IF NOT EXISTS idx_stress_user_timestamp ON stress_history(user_id, timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_stress_session ON stress_history(session_id)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_user ON user_recommendations(user_id, created_at)',
            'CREATE INDEX IF NOT EXISTS idx_calibration_user ON user_calibration(user_id, is_active)',
            'CREATE INDEX IF NOT EXISTS idx_goals_user ON user_goals(user_id, achieved)',
            'CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON user_notifications(user_id, read_at)'
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        print("✅ Database tables created successfully!")

def create_sample_user():
    """Create a sample user for testing"""
    print("Creating sample user for testing...")
    
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if sample user already exists
            cursor.execute('SELECT id FROM users WHERE username = ?', ('demo_user',))
            if cursor.fetchone():
                print("Sample user already exists.")
                return
            
            # Create sample user
            user_id = secrets.token_urlsafe(16)
            password_hash = bcrypt.hashpw(b'demo123456', bcrypt.gensalt()).decode('utf-8')
            
            profile_data = {
                'full_name': 'Demo User',
                'bio': 'Sample user for AI Wellness Assistant testing',
                'registration_date': datetime.now().isoformat(),
                'avatar_url': '',
                'wellness_goals': ['Reduce daily stress', 'Improve work-life balance'],
                'emergency_contact': ''
            }
            
            preferences = {
                'stress_thresholds': {
                    'low': 0.25,
                    'moderate': 0.55,
                    'high': 0.75
                },
                'notifications': {
                    'stress_alerts': True,
                    'daily_summary': True,
                    'wellness_tips': True
                },
                'privacy': {
                    'data_retention_days': 90,
                    'share_anonymous_data': False
                },
                'interface': {
                    'theme': 'light',
                    'language': 'en',
                    'timezone': 'UTC'
                }
            }
            
            cursor.execute('''
                INSERT INTO users (id, username, email, password_hash, profile_data, preferences)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id, 'demo_user', 'demo@wellness.ai', password_hash,
                json.dumps(profile_data), json.dumps(preferences)
            ))
            
            # Create sample wellness goals
            goals = [
                {
                    'goal_type': 'daily_stress_avg',
                    'goal_value': 0.4,
                    'description': 'Keep daily average stress below 0.4',
                    'target_date': (datetime.now() + timedelta(days=30)).date().isoformat()
                },
                {
                    'goal_type': 'weekly_sessions',
                    'goal_value': 5.0,
                    'description': 'Complete 5 wellness monitoring sessions per week',
                    'target_date': (datetime.now() + timedelta(days=7)).date().isoformat()
                }
            ]
            
            for goal in goals:
                goal_id = secrets.token_urlsafe(16)
                cursor.execute('''
                    INSERT INTO user_goals (id, user_id, goal_type, goal_value, target_date, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    goal_id, user_id, goal['goal_type'], goal['goal_value'],
                    goal['target_date'], goal['description']
                ))
            
            # Create sample notifications
            notifications = [
                {
                    'title': 'Welcome to AI Wellness Assistant!',
                    'message': 'Start your first monitoring session to get personalized wellness insights.',
                    'type': 'success'
                },
                {
                    'title': 'Tip: Calibrate Your Baseline',
                    'message': 'Take a few calm monitoring sessions to help us learn your personal stress patterns.',
                    'type': 'info'
                }
            ]
            
            for notification in notifications:
                notif_id = secrets.token_urlsafe(16)
                cursor.execute('''
                    INSERT INTO user_notifications (id, user_id, title, message, type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    notif_id, user_id, notification['title'],
                    notification['message'], notification['type']
                ))
            
            conn.commit()
            print(f"✅ Sample user created successfully!")
            print(f"   Username: demo_user")
            print(f"   Password: demo123456")
            print(f"   Email: demo@wellness.ai")
            
    except Exception as e:
        print(f"❌ Error creating sample user: {e}")

def create_sample_data():
    """Create sample wellness data for demonstration"""
    print("Creating sample wellness data...")
    
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Get sample user
            cursor.execute('SELECT id FROM users WHERE username = ?', ('demo_user',))
            result = cursor.fetchone()
            if not result:
                print("Sample user not found. Create sample user first.")
                return
            
            user_id = result[0]
            
            # Create sample session
            session_id = secrets.token_urlsafe(16)
            session_start = datetime.now() - timedelta(hours=2)
            session_end = session_start + timedelta(minutes=15)
            
            session_data = {
                'session_id': session_id,
                'start_time': session_start.isoformat(),
                'end_time': session_end.isoformat(),
                'user_id': user_id,
                'device_info': 'WebCam - Demo Session'
            }
            
            cursor.execute('''
                INSERT INTO user_sessions 
                (id, user_id, session_data, created_at, duration, avg_stress_score, 
                 max_stress_score, min_stress_score, stress_episodes, face_detection_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, user_id, json.dumps(session_data),
                session_start, 900,  # 15 minutes
                0.35, 0.68, 0.12, 3, 0.95
            ))
            
            # Create sample stress readings
            stress_readings = [
                (0.15, 'low'), (0.22, 'low'), (0.28, 'low'),
                (0.35, 'moderate'), (0.42, 'moderate'), (0.38, 'moderate'),
                (0.51, 'moderate'), (0.63, 'high'), (0.68, 'high'),
                (0.45, 'moderate'), (0.32, 'moderate'), (0.25, 'low'),
                (0.19, 'low'), (0.12, 'low')
            ]
            
            for i, (score, level) in enumerate(stress_readings):
                reading_id = secrets.token_urlsafe(16)
                timestamp = session_start + timedelta(minutes=i)
                
                cursor.execute('''
                    INSERT INTO stress_history 
                    (id, user_id, session_id, timestamp, stress_score, raw_stress_score,
                     stress_level, face_detected, confidence, user_calibrated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    reading_id, user_id, session_id, timestamp,
                    score, score + 0.02, level, True, 0.85, False
                ))
            
            # Create user calibration data
            calib_id = secrets.token_urlsafe(16)
            calibration_readings = [0.18, 0.22, 0.25, 0.20, 0.24]  # Low stress baseline
            
            cursor.execute('''
                INSERT INTO user_calibration 
                (id, user_id, baseline_stress, calibration_readings, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                calib_id, user_id, 0.22, json.dumps(calibration_readings),
                'Initial baseline calibration during relaxed state'
            ))
            
            conn.commit()
            print("✅ Sample wellness data created successfully!")
            
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")

def verify_database():
    """Verify database integrity and display statistics"""
    print("Verifying database...")
    
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Get table statistics
            tables = [
                'users', 'user_sessions', 'stress_history', 
                'user_recommendations', 'user_calibration', 
                'user_goals', 'user_notifications'
            ]
            
            print("\n📊 Database Statistics:")
            print("-" * 40)
            
            for table in tables:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    print(f"📋 {table:20} {count:6} records")
                except sqlite3.OperationalError:
                    print(f"⚠️  {table:20}     Table not found")
            
            # Get specific user information
            cursor.execute('SELECT username, email, created_at, last_login FROM users WHERE is_active = TRUE')
            users = cursor.fetchall()
            
            if users:
                print("\n👥 Active Users:")
                print("-" * 40)
                for username, email, created, last_login in users:
                    print(f"👤 {username} ({email})")
                    print(f"   Created: {created}")
                    print(f"   Last login: {last_login or 'Never'}")
            
            print("\n✅ Database verification complete!")
            
    except Exception as e:
        print(f"❌ Error verifying database: {e}")

def reset_database():
    """Reset database (WARNING: Deletes all data)"""
    response = input("⚠️  WARNING: This will delete ALL user data. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Database reset cancelled.")
        return
    
    try:
        if os.path.exists(DATABASE_PATH):
            os.remove(DATABASE_PATH)
            print("🗑️  Database file deleted.")
        
        create_database_tables()
        print("✅ Database reset complete!")
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")

def main():
    """Main setup function"""
    print("🧠 AI Wellness Assistant - Database Setup")
    print("=" * 50)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Create/Initialize database tables")
        print("2. Create sample user and data")
        print("3. Verify database integrity")
        print("4. Reset database (DELETE ALL DATA)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            create_database_tables()
        elif choice == '2':
            create_sample_user()
            create_sample_data()
        elif choice == '3':
            verify_database()
        elif choice == '4':
            reset_database()
        elif choice == '5':
            print("👋 Database setup complete. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == '__main__':
    main()