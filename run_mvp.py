#!/usr/bin/env python3
"""
AI Wellness Assistant - MVP Startup Script
Author: Pricop George
Description: Simple script to start the MVP application with proper setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'flask_cors', 
        'cv2',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_mvp.txt")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Directory '{directory}' ready")

def check_camera_access():
    """Test camera access"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera access verified")
                return True
            else:
                print("⚠️ Warning: Camera detected but cannot read frames")
                return True  # Still allow startup
        else:
            print("⚠️ Warning: Cannot access camera (index 0)")
            print("  - Check if camera is connected")
            print("  - Try different camera index in config.py")
            return True  # Allow startup anyway
    except Exception as e:
        print(f"⚠️ Warning: Camera test failed: {e}")
        return True  # Allow startup anyway

def start_application():
    """Start the MVP application"""
    print("
🚀 Starting AI Wellness Assistant MVP...")
    
    try:
        # Import and run the app
        from app_mvp import app, logger, APP_NAME, VERSION, FLASK_HOST, FLASK_PORT
        
        print(f"✓ {APP_NAME} v{VERSION} loaded successfully")
        print(f"✓ Server starting on http://{FLASK_HOST}:{FLASK_PORT}")
        print(f"✓ Open your browser and navigate to: http://localhost:{FLASK_PORT}")
        print("\n🔍 Logs will be saved to: logs/app.log")
        print("💾 Session data will be saved to: data/")
        print("\n⏹️  Press Ctrl+C to stop the application\n")
        
        app.run(
            debug=False,  # Set to False for production-like behavior
            host=FLASK_HOST,
            port=FLASK_PORT,
            threaded=True  # Enable threading for better performance
        )
        
    except ImportError as e:
        print(f"❌ Error importing application modules: {e}")
        print("Make sure all files are present and dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🧠 AI Wellness Assistant - MVP Startup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    print("✓ Python version compatible")
    
    # Setup directories
    setup_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 To install dependencies, run:")
        print("pip install -r requirements_mvp.txt")
        sys.exit(1)
    print("✓ All dependencies installed")
    
    # Check camera access
    check_camera_access()
    
    print("\n🟢 All checks passed! Starting application...")
    print("=" * 50)
    
    # Start the application
    if not start_application():
        sys.exit(1)

if __name__ == "__main__":
    main()
