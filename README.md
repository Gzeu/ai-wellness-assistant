# 🧠 AI Wellness Assistant

**Real-time stress detection system using facial expressions analysis and computer vision**

![Version](https://img.shields.io/badge/version-1.0.0--MVP-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-MVP%20Ready-success)

## 📝 Overview

AI Wellness Assistant is an advanced stress detection system that uses computer vision and facial expression analysis to monitor wellness in real-time. The application provides personalized wellness recommendations based on detected stress levels, helping users maintain better mental health.

### ✨ Key Features

- ✅ **Real-time face detection** using OpenCV Haar Cascades
- ✅ **Advanced stress analysis** from facial micro-expressions
- ✅ **Live video streaming** with stress indicators overlay
- ✅ **Personalized wellness recommendations** based on stress levels
- ✅ **Session data persistence** for tracking progress over time
- ✅ **Responsive web interface** with real-time updates
- ✅ **RESTful API** for integration with other systems
- ✅ **Comprehensive logging** and error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera access
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Gzeu/ai-wellness-assistant.git
cd ai-wellness-assistant
```

2. **Install dependencies:**
```bash
# For MVP version (recommended)
pip install -r requirements_mvp.txt

# OR for full version (future development)
pip install -r requirements.txt
```

3. **Run the application:**
```bash
# Quick start with MVP
python run_mvp.py

# OR direct execution
python app_mvp.py
```

4. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - Allow camera access when prompted
   - Start monitoring your wellness!

## 🎯 Features Deep Dive

### Stress Detection Engine

Our advanced stress detection system analyzes:
- **Facial landmarks** for micro-expression detection
- **Eye region analysis** for stress indicators
- **Mouth and jaw tension** patterns
- **Overall facial symmetry** changes

### Stress Levels Classification

- 🟢 **Low** (0.0 - 0.3): Relaxed and calm state
- 🟠 **Moderate** (0.3 - 0.6): Some tension detected
- 🔴 **High** (0.6 - 0.8): Significant stress present
- ⚫ **Very High** (0.8 - 1.0): Immediate attention needed

### Wellness Recommendations

Based on detected stress levels, the system provides:
- **Breathing exercises** for moderate stress
- **Progressive muscle relaxation** techniques
- **Mindfulness practices** for high stress levels
- **Emergency wellness interventions** for critical levels

## 📊 Project Structure

```
ai-wellness-assistant/
├── app_mvp.py              # MVP main application
├── config.py               # Configuration settings
├── requirements_mvp.txt    # MVP dependencies
├── run_mvp.py              # Quick start script
├── templates/
│   └── index_mvp.html      # Responsive web interface
├── data/                   # Session storage
├── logs/                   # Application logs
├── static/                 # CSS, JS, images
└── README.md               # This documentation
```

## 🔧 Configuration

Customize your experience by editing `config.py`:

```python
# Camera settings
WEBCAM_INDEX = 0           # Camera index (0 = default)
CAMERA_WIDTH = 640         # Video resolution width
CAMERA_HEIGHT = 480        # Video resolution height

# Stress detection thresholds
STRESS_THRESHOLD_LOW = 0.3
STRESS_THRESHOLD_MODERATE = 0.6
STRESS_THRESHOLD_HIGH = 0.8

# Server configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DEBUG_MODE = False
```

## 🎮 Usage Guide

### Starting Your Wellness Session

1. **Launch Application**: Open browser to `http://localhost:5000`
2. **Camera Setup**: Allow camera access and position your face clearly
3. **Start Monitoring**: Click "Start Monitoring" button
4. **Real-time Analysis**: Watch your stress levels update in real-time
5. **Follow Recommendations**: Use provided wellness tips based on your stress state

### Understanding the Interface

- **Left Panel**: Live camera feed with face detection overlay
- **Right Panel**: 
  - Real-time stress metrics and scores
  - Color-coded stress level indicators
  - Personalized wellness recommendations
  - Session statistics and history

## 📋 API Reference

### Core Endpoints

```bash
# Start stress monitoring
POST /start_monitoring

# Stop monitoring session
POST /stop_monitoring

# Get current stress data
GET /get_stress_data
Response: {"score": 0.45, "level": "moderate", "timestamp": "..."}

# Get wellness recommendations
GET /get_recommendations
Response: {"recommendations": [...], "level": "moderate"}

# Live video stream
GET /video_feed

# Application health check
GET /health
```

### Data Management

```bash
# Save current session
POST /save_session

# Retrieve session history
GET /get_sessions

# Export session data
GET /export_sessions?format=json
```

## 🧪 Testing & Validation

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/ -v --cov=app_mvp

# Run specific test categories
pytest tests/test_stress_detection.py -v
pytest tests/test_api_endpoints.py -v
```

### Manual Testing Checklist

- [ ] Camera detection and initialization
- [ ] Face detection accuracy in various lighting
- [ ] Stress level calculation consistency
- [ ] Real-time video streaming performance
- [ ] Wellness recommendations relevance
- [ ] Session data persistence
- [ ] Cross-browser compatibility
- [ ] Mobile responsiveness

## 🐛 Troubleshooting

### Common Issues

**📷 Camera not detected:**
```bash
# Check camera permissions
# Try different camera indices
WEBCAM_INDEX = 1  # or 2, 3...

# Verify camera availability
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**🎯 Poor stress detection:**
- Ensure good lighting conditions
- Position face clearly within camera frame
- Avoid excessive movement during analysis
- Check that face detection box appears

**⚡ Performance issues:**
```python
# Reduce resolution for better performance
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360

# Adjust processing frequency
PROCESSING_FPS = 15  # Lower FPS for slower systems
```

**📊 Application crashes:**
- Check logs in `logs/app.log`
- Verify all dependencies installed correctly
- Ensure Python version compatibility (3.8+)

## 🚀 Development Roadmap

### Current Version (MVP v1.0.0)
- [x] Real-time face detection
- [x] Basic stress level calculation
- [x] Web interface with live streaming
- [x] Session data persistence
- [x] Wellness recommendations

### Upcoming Features (v2.0.0)
- [ ] **Advanced FACS Analysis** - Facial Action Coding System integration
- [ ] **Machine Learning Models** - AI-powered emotion recognition
- [ ] **Historical Analytics** - Trend analysis and insights
- [ ] **Mobile App** - iOS and Android applications
- [ ] **Cloud Integration** - Data synchronization and backup
- [ ] **Multi-user Support** - User profiles and authentication
- [ ] **Wearable Integration** - Heart rate and biometric data
- [ ] **Advanced Coaching** - Personalized wellness programs

### Future Enhancements (v3.0.0+)
- [ ] **Real-time Emotion Detection** - Beyond stress analysis
- [ ] **Voice Analysis** - Speech pattern stress indicators
- [ ] **Environmental Factors** - Light, noise impact analysis
- [ ] **Team Wellness** - Corporate wellness dashboards
- [ ] **Research Tools** - Academic and clinical research features

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/ai-wellness-assistant.git
cd ai-wellness-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements_dev.txt

# Create feature branch
git checkout -b feature/amazing-feature
```

### Contribution Guidelines

1. **Code Quality**: Follow PEP 8 style guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and code comments
4. **Commit Messages**: Use clear, descriptive commit messages
5. **Pull Requests**: Include detailed description of changes

### Areas We Need Help

- 🔬 **Algorithm Improvement**: Better stress detection algorithms
- 🎨 **UI/UX Design**: Enhanced user interface design
- 📱 **Mobile Development**: iOS and Android apps
- 📊 **Data Science**: Analytics and insights features
- 📝 **Documentation**: Tutorials and guides
- 🌍 **Internationalization**: Multi-language support

## 📜 Documentation

### Additional Resources

- **[API Documentation](docs/api.md)** - Complete API reference
- **[Development Guide](docs/development.md)** - Setup and contribution guide
- **[Deployment Guide](docs/deployment.md)** - Production deployment
- **[Research Papers](docs/research.md)** - Scientific background and references

### Research & Background

This project is based on established research in:
- **Computer Vision** - OpenCV and facial landmark detection
- **Psychology** - Facial Action Coding System (FACS)
- **Stress Research** - Physiological stress indicators
- **Machine Learning** - Pattern recognition and classification

## 🔒 Security & Privacy

### Data Privacy
- **Local Processing**: All analysis happens locally on your device
- **No Cloud Upload**: Video data never leaves your computer
- **Session Data**: Only statistical data is stored locally
- **User Control**: Complete control over data deletion

### Security Features
- **Secure Connections**: HTTPS support for production
- **Data Encryption**: Local data encryption options
- **Access Controls**: Camera permission management
- **Audit Logging**: Complete activity logging

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-party Licenses
- **OpenCV**: BSD 3-Clause License
- **Flask**: BSD 3-Clause License
- **NumPy**: BSD License

## 👨‍💻 Author & Maintainer

**George Pricop**
- 👁️ GitHub: [@Gzeu](https://github.com/Gzeu)
- 📧 Email: pricopgeorge@gmail.com
- 💼 LinkedIn: [George Pricop](https://linkedin.com/in/george-pricop)
- 🐦 Twitter: [@GeorgePricop](https://twitter.com/GeorgePricop)

## 🙏 Acknowledgments

### Technology Stack
- **OpenCV Community** - Computer vision tools and algorithms
- **Flask Team** - Web framework for Python
- **NumPy Contributors** - Numerical computing library
- **Python Software Foundation** - Python programming language

### Research Contributors
- **Paul Ekman** - Facial Action Coding System (FACS)
- **Computer Vision Research Community** - Facial landmark detection
- **Stress Research Scientists** - Physiological stress indicators

### Special Thanks
- **Beta Testers** - Early feedback and testing
- **Open Source Community** - Libraries and tools
- **Mental Health Advocates** - Wellness and stress management insights

## 📈 Version History

### v1.0.0-MVP (September 2025)
- ✨ Initial MVP release
- ✅ Real-time face detection with OpenCV
- ✅ Basic stress level calculation
- ✅ Responsive web interface
- ✅ Session data persistence
- ✅ Wellness recommendations system
- ✅ RESTful API endpoints
- ✅ Comprehensive documentation

### v0.1.0-Alpha (September 2025)
- 🛠️ Initial development setup
- 📋 Project structure and dependencies
- 📝 Basic README and documentation

---

## 🏆 Project Status

**Current Status**: 🚀 **MVP Ready for Testing**

The AI Wellness Assistant MVP is fully functional and ready for:
- ✅ User testing and feedback collection
- ✅ Demo presentations and showcases
- ✅ Development team review and analysis
- ✅ Performance benchmarking and optimization
- ✅ Feature validation and user experience testing

**Next Milestone**: Community feedback integration and v2.0.0 planning

---

<div align="center">

**⚡ Ready to monitor your wellness in real-time!**

*Start your journey towards better stress management with AI-powered insights.*

[![GitHub Stars](https://img.shields.io/github/stars/Gzeu/ai-wellness-assistant?style=social)](https://github.com/Gzeu/ai-wellness-assistant/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Gzeu/ai-wellness-assistant?style=social)](https://github.com/Gzeu/ai-wellness-assistant/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Gzeu/ai-wellness-assistant)](https://github.com/Gzeu/ai-wellness-assistant/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/Gzeu/ai-wellness-assistant)](https://github.com/Gzeu/ai-wellness-assistant/pulls)

</div>