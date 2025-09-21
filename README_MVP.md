# 🧠 AI Wellness Assistant - MVP Version

**Real-time stress detection system using facial expressions analysis**

![Version](https://img.shields.io/badge/version-0.1.0--MVP-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Overview

AI Wellness Assistant MVP is a simplified version of an advanced stress detection system that uses computer vision to analyze facial expressions in real-time. This MVP provides core functionality for:

- ✅ **Real-time face detection** using OpenCV
- ✅ **Basic stress level calculation** from facial features  
- ✅ **Live video streaming** with stress indicators overlay
- ✅ **Wellness recommendations** based on detected stress levels
- ✅ **Session data persistence** for tracking progress
- ✅ **Responsive web interface** for easy interaction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera access
- Modern web browser

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Gzeu/ai-wellness-assistant.git
cd ai-wellness-assistant
```

2. **Switch to MVP branch:**
```bash
git checkout mvp-development
```

3. **Install dependencies:**
```bash
pip install -r requirements_mvp.txt
```

4. **Run the MVP application:**
```bash
python app_mvp.py
```

5. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - Allow camera access when prompted

## 🎯 MVP Features

### Core Functionality
- **Face Detection**: Automatic face detection using OpenCV Haar Cascades
- **Stress Analysis**: Basic stress scoring based on facial region analysis
- **Real-time Display**: Live stress levels with color-coded indicators
- **Recommendations**: Personalized wellness tips based on current stress state

### Stress Levels
- 🟢 **Low** (0.0 - 0.3): Relaxed state
- 🟠 **Moderate** (0.3 - 0.6): Some tension present  
- 🔴 **High** (0.6 - 0.8): Significant stress detected
- ⚫ **Very High** (0.8 - 1.0): Immediate attention needed

### Wellness Recommendations
The system provides contextual recommendations:
- **Breathing exercises** for moderate stress
- **Progressive muscle relaxation** techniques
- **Mindfulness practices** for high stress levels
- **Emergency interventions** for very high stress

## 📁 Project Structure

```
ai-wellness-assistant/
├── app_mvp.py              # Main MVP application
├── config.py               # Configuration settings
├── requirements_mvp.txt    # MVP dependencies
├── templates/
│   └── index_mvp.html     # MVP web interface
├── data/                  # Session data storage
├── logs/                  # Application logs
└── README_MVP.md          # This file
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Camera settings
WEBCAM_INDEX = 0           # Camera index (0 = default)
CAMERA_WIDTH = 640         # Video width
CAMERA_HEIGHT = 480        # Video height

# Stress thresholds
STRESS_THRESHOLD_LOW = 0.3
STRESS_THRESHOLD_MODERATE = 0.6  
STRESS_THRESHOLD_HIGH = 0.8

# Server settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
```

## 🎮 Usage Instructions

### Starting Monitoring
1. Open the web interface at `http://localhost:5000`
2. Click **"Start Monitoring"** button
3. Allow camera access when prompted
4. Position your face in the camera frame
5. Watch real-time stress levels update

### Understanding the Interface
- **Left Panel**: Live camera feed with face detection overlay
- **Right Panel**: Stress metrics and wellness recommendations
- **Color Indicators**: Green (low) → Orange (moderate) → Red (high stress)

### Session Management
- Sessions are automatically saved when monitoring stops
- View session history via the API endpoints
- Data is stored in JSON format in the `data/` directory

## 📊 API Endpoints

The MVP includes RESTful API endpoints:

### Core Endpoints
- `GET /` - Main web interface
- `POST /start_monitoring` - Start camera and stress detection
- `POST /stop_monitoring` - Stop monitoring session
- `GET /get_stress_data` - Get current stress metrics
- `GET /get_recommendations` - Get wellness recommendations
- `GET /video_feed` - Live video stream

### Data Management
- `POST /save_session` - Save current session data
- `GET /get_sessions` - Retrieve saved sessions
- `GET /health` - Application health check

### Example API Usage

```javascript
// Start monitoring
fetch('/start_monitoring', {method: 'POST'})
  .then(response => response.json())
  .then(data => console.log(data));

// Get stress data
fetch('/get_stress_data')
  .then(response => response.json())
  .then(data => console.log('Stress:', data.score));
```

## 🧪 Testing

Run basic tests:
```bash
pytest tests/ -v
```

Check application health:
```bash
curl http://localhost:5000/health
```

## 🐛 Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions in browser
- Try different `WEBCAM_INDEX` values (0, 1, 2...)
- Ensure no other applications are using the camera

**Poor stress detection:**
- Ensure good lighting conditions
- Position face clearly in camera frame  
- Avoid excessive movement during analysis
- Check that face detection box appears around your face

**Application crashes:**
- Check logs in `logs/app.log`
- Verify all dependencies are installed
- Try restarting the application

### Performance Optimization

- **Lower resolution**: Reduce `CAMERA_WIDTH/HEIGHT` for better performance
- **Adjust FPS**: Modify frame rate in config for smoother operation
- **Close other applications**: Free up system resources

## 🔮 Future Development

### Planned Features (Post-MVP)
- [ ] Advanced FACS (Facial Action Coding System) analysis
- [ ] Machine learning model integration
- [ ] Historical trend analysis and insights
- [ ] Mobile app development
- [ ] Integration with wearable devices
- [ ] Multi-user support with profiles
- [ ] Advanced wellness coaching algorithms

### Technical Improvements
- [ ] Real-time emotion detection
- [ ] Better stress calculation algorithms  
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] User authentication system
- [ ] Data export features (CSV, PDF reports)
- [ ] Cloud deployment ready

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**George Pricop**
- GitHub: [@Gzeu](https://github.com/Gzeu)
- Email: pricopgeorge@gmail.com

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Flask framework for web development
- Contributors and testers

## 📈 Version History

- **v0.1.0-MVP** (September 2025)
  - Initial MVP release
  - Basic face detection and stress analysis
  - Real-time web interface
  - Session data persistence
  - Wellness recommendations system

---

**⚡ Ready to monitor your wellness in real-time!**

*Start your journey towards better stress management with AI-powered insights.*