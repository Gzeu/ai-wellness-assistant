#!/usr/bin/env python3
"""
Authentication Routes for AI Wellness Assistant
User registration, login, profile management, and dashboard endpoints

Author: George Pricop
Version: 2.0.0-AUTH
Date: September 2025
"""

import json
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash

# Import authentication components (will be created)
try:
    from auth import auth_manager, PersonalizedDataManager, require_auth
except ImportError:
    # Fallback for development
    class MockAuthManager:
        def authenticate_user(self, username, password):
            return False, None, "Auth system not available"
        def create_user(self, username, email, password, profile=None):
            return False, "Auth system not available"
    
    auth_manager = MockAuthManager()
    PersonalizedDataManager = None
    
    def require_auth(f):
        return f

# Create authentication blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration endpoint"""
    if request.method == 'GET':
        return render_template('auth/register.html')
    
    try:
        data = request.get_json() or request.form
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        full_name = data.get('full_name', '').strip()
        
        # Validation
        errors = []
        if not username:
            errors.append('Username is required')
        elif len(username) < 3:
            errors.append('Username must be at least 3 characters long')
        
        if not email or '@' not in email:
            errors.append('Valid email is required')
        
        if not password:
            errors.append('Password is required')
        elif len(password) < 8:
            errors.append('Password must be at least 8 characters long')
        
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        
        # Create user profile data
        profile_data = {
            'full_name': full_name,
            'registration_date': datetime.now().isoformat(),
            'avatar_url': '',
            'bio': '',
            'wellness_goals': [],
            'emergency_contact': ''
        }
        
        # Attempt to create user
        success, message = auth_manager.create_user(username, email, password, profile_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Registration successful! Please log in.',
                'redirect': '/auth/login'
            })
        else:
            return jsonify({
                'success': False,
                'errors': [message]
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'errors': [f'Registration error: {str(e)}']
        }), 500

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login endpoint"""
    if request.method == 'GET':
        return render_template('auth/login.html')
    
    try:
        data = request.get_json() or request.form
        username = data.get('username', '').strip()
        password = data.get('password', '')
        remember_me = data.get('remember_me', False)
        
        if not username or not password:
            return jsonify({
                'success': False,
                'errors': ['Username and password are required']
            }), 400
        
        # Authenticate user
        success, user, message = auth_manager.authenticate_user(username, password)
        
        if success and user:
            login_user(user, remember=remember_me)
            
            # Update session data
            session['user_id'] = user.id
            session['username'] = user.username
            session['login_time'] = datetime.now().isoformat()
            
            return jsonify({
                'success': True,
                'message': 'Login successful!',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email
                },
                'redirect': '/dashboard'
            })
        else:
            return jsonify({
                'success': False,
                'errors': [message]
            }), 401
    
    except Exception as e:
        return jsonify({
            'success': False,
            'errors': [f'Login error: {str(e)}']
        }), 500

@auth_bp.route('/logout', methods=['POST', 'GET'])
@login_required
def logout():
    """User logout endpoint"""
    try:
        user_id = current_user.id if current_user.is_authenticated else None
        
        logout_user()
        session.clear()
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully',
            'redirect': '/'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Logout error: {str(e)}'
        }), 500

@auth_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile management"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'user': current_user.to_dict()
        })
    
    try:
        data = request.get_json() or request.form
        
        # Update profile data
        profile_updates = {}
        preferences_updates = {}
        
        # Profile fields
        profile_fields = ['full_name', 'bio', 'avatar_url', 'wellness_goals', 'emergency_contact']
        for field in profile_fields:
            if field in data:
                profile_updates[field] = data[field]
        
        # Preferences fields
        if 'preferences' in data:
            preferences_updates = data['preferences']
        
        # Update user data
        success, message = auth_manager.update_user_profile(
            current_user.id, 
            profile_updates if profile_updates else None,
            preferences_updates if preferences_updates else None
        )
        
        if success:
            # Reload user data
            updated_user = auth_manager.load_user(current_user.id)
            return jsonify({
                'success': True,
                'message': message,
                'user': updated_user.to_dict() if updated_user else current_user.to_dict()
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Profile update error: {str(e)}'
        }), 500

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    try:
        data = request.get_json() or request.form
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not current_password:
            return jsonify({
                'success': False,
                'error': 'Current password is required'
            }), 400
        
        if not new_password or len(new_password) < 8:
            return jsonify({
                'success': False,
                'error': 'New password must be at least 8 characters long'
            }), 400
        
        if new_password != confirm_password:
            return jsonify({
                'success': False,
                'error': 'New passwords do not match'
            }), 400
        
        # Change password
        success, message = auth_manager.change_password(
            current_user.id, current_password, new_password
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Password change error: {str(e)}'
        }), 500

@auth_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with wellness analytics"""
    try:
        # Get user analytics if PersonalizedDataManager is available
        analytics = {}
        recommendations = []
        
        if PersonalizedDataManager:
            analytics = PersonalizedDataManager.get_user_analytics(current_user.id, 30)
            recommendations = PersonalizedDataManager.get_personalized_recommendations(current_user.id)
        
        return jsonify({
            'success': True,
            'user': current_user.to_dict(),
            'analytics': analytics,
            'recommendations': recommendations,
            'dashboard_data': {
                'total_sessions': analytics.get('total_sessions', 0),
                'avg_stress_level': analytics.get('average_stress', 0.0),
                'stress_trend': analytics.get('stress_trend', []),
                'last_session': analytics.get('last_session_date', None)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Dashboard error: {str(e)}'
        }), 500

@auth_bp.route('/analytics/<int:days>')
@login_required
def get_analytics(days):
    """Get user wellness analytics for specified period"""
    try:
        if not PersonalizedDataManager:
            return jsonify({
                'success': False,
                'error': 'Analytics not available'
            }), 503
        
        # Validate days parameter
        if days < 1 or days > 365:
            days = 30  # Default to 30 days
        
        analytics = PersonalizedDataManager.get_user_analytics(current_user.id, days)
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'period_days': days
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analytics error: {str(e)}'
        }), 500

@auth_bp.route('/save-session', methods=['POST'])
@login_required
def save_wellness_session():
    """Save wellness monitoring session data"""
    try:
        if not PersonalizedDataManager:
            return jsonify({
                'success': False,
                'error': 'Session storage not available'
            }), 503
        
        data = request.get_json() or request.form
        session_data = data.get('session_data', {})
        analytics = data.get('analytics', {})
        
        # Validate session data
        required_fields = ['start_time', 'end_time', 'stress_readings']
        for field in required_fields:
            if field not in session_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        success, session_id = PersonalizedDataManager.save_user_session(
            current_user.id, session_data, analytics
        )
        
        if success:
            # Save individual stress data points
            stress_readings = session_data.get('stress_readings', [])
            for reading in stress_readings:
                PersonalizedDataManager.add_stress_data_point(
                    current_user.id,
                    reading.get('score', 0.0),
                    reading.get('level', 'unknown'),
                    reading.get('face_detected', False),
                    session_id
                )
            
            return jsonify({
                'success': True,
                'message': 'Session saved successfully',
                'session_id': session_id
            })
        else:
            return jsonify({
                'success': False,
                'error': session_id  # session_id contains error message in this case
            }), 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Session save error: {str(e)}'
        }), 500

@auth_bp.route('/recommendations')
@login_required
def get_recommendations():
    """Get personalized wellness recommendations"""
    try:
        if not PersonalizedDataManager:
            return jsonify({
                'success': False,
                'error': 'Recommendations not available'
            }), 503
        
        recommendations = PersonalizedDataManager.get_personalized_recommendations(current_user.id)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Recommendations error: {str(e)}'
        }), 500

@auth_bp.route('/status')
def auth_status():
    """Check authentication status"""
    try:
        if current_user.is_authenticated:
            return jsonify({
                'authenticated': True,
                'user': {
                    'id': current_user.id,
                    'username': current_user.username,
                    'email': current_user.email
                }
            })
        else:
            return jsonify({
                'authenticated': False,
                'user': None
            })
    
    except Exception as e:
        return jsonify({
            'authenticated': False,
            'error': f'Status check error: {str(e)}'
        }), 500

# Error handlers for authentication blueprint
@auth_bp.errorhandler(401)
def unauthorized(error):
    return jsonify({
        'success': False,
        'error': 'Unauthorized access',
        'message': 'Please log in to access this resource'
    }), 401

@auth_bp.errorhandler(403)
def forbidden(error):
    return jsonify({
        'success': False,
        'error': 'Forbidden',
        'message': 'You do not have permission to access this resource'
    }), 403

@auth_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# Export blueprint
__all__ = ['auth_bp']