/**
 * AI Wellness Assistant - Webcam Integration
 * Handles webcam access, real-time monitoring, and UI updates
 */

let isMonitoring = false;
let webcamStream = null;
let stressUpdateInterval = null;
let currentStressScore = 0;

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const webcam = document.getElementById('webcam');
const analysisStatus = document.getElementById('analysisStatus');
const stressOverlay = document.getElementById('stressOverlay');

// Initialize webcam
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            } 
        });
        webcam.srcObject = webcamStream;
        return true;
    } catch (err) {
        console.error('Error accessing webcam:', err);
        updateAnalysisStatus(
            'Unable to access webcam. Please ensure camera permissions are granted.',
            'error'
        );
        return false;
    }
}

// Start monitoring
async function startMonitoring() {
    if (await startWebcam()) {
        isMonitoring = true;
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        stressOverlay.style.display = 'block';
        
        updateAnalysisStatus('Real-time stress analysis active', 'success');

        // Start backend monitoring
        try {
            await fetch('/start_monitoring', { method: 'POST' });
            
            // Update stress data every 2 seconds
            stressUpdateInterval = setInterval(updateStressData, 2000);
        } catch (error) {
            console.error('Error starting backend monitoring:', error);
            updateAnalysisStatus('Error starting analysis. Please try again.', 'error');
        }
    }
}

// Stop monitoring
async function stopMonitoring() {
    isMonitoring = false;
    startBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    stressOverlay.style.display = 'none';
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    if (stressUpdateInterval) {
        clearInterval(stressUpdateInterval);
        stressUpdateInterval = null;
    }
    
    updateAnalysisStatus('Monitoring stopped. Click "Start Monitoring" to resume.', 'info');

    try {
        await fetch('/stop_monitoring', { method: 'POST' });
    } catch (error) {
        console.error('Error stopping backend monitoring:', error);
    }
}

// Update stress data from backend
async function updateStressData() {
    try {
        const response = await fetch('/get_stress_data');
        const data = await response.json();
        
        currentStressScore = data.current_score;
        updateStressDisplay(data.current_score);
        
        // Show recommendations for high stress
        if (data.current_score > 0.6) {
            showRecommendations();
        }
        
        // Update trend data (could be used for mini-charts)
        if (data.trend && data.trend.length > 0) {
            updateTrendVisualization(data.trend);
        }
        
    } catch (err) {
        console.error('Error updating stress data:', err);
        updateAnalysisStatus('Connection error. Retrying...', 'warning');
    }
}

// Update stress display elements
function updateStressDisplay(stressScore) {
    const percentage = Math.round(stressScore * 100);
    const stressLevel = getStressLevel(stressScore);
    
    // Update overlay
    const stressLevelEl = document.getElementById('stressLevel');
    const stressStatusEl = document.getElementById('stressStatus');
    if (stressLevelEl) stressLevelEl.textContent = percentage + '%';
    if (stressStatusEl) stressStatusEl.textContent = stressLevel.level;
    
    // Update main meter
    const stressPercentageEl = document.getElementById('stressPercentage');
    const stressLabelEl = document.getElementById('stressLabel');
    if (stressPercentageEl) stressPercentageEl.textContent = percentage + '%';
    if (stressLabelEl) stressLabelEl.textContent = stressLevel.level;
    
    // Update meter colors
    const meterElement = document.getElementById('stressMeter');
    if (meterElement) {
        meterElement.style.filter = `hue-rotate(${Math.min(stressScore * 120, 120)}deg)`;
        
        // Add pulse effect for high stress
        if (stressScore > 0.7) {
            meterElement.classList.add('pulse-animation');
        } else {
            meterElement.classList.remove('pulse-animation');
        }
    }
}

// Get stress level interpretation
function getStressLevel(score) {
    if (score < 0.2) return { level: 'Relaxed', color: '#22c55e' };
    if (score < 0.4) return { level: 'Slightly Elevated', color: '#8bc34a' };
    if (score < 0.6) return { level: 'Moderate Stress', color: '#f59e0b' };
    if (score < 0.8) return { level: 'High Stress', color: '#ff9800' };
    return { level: 'Very High Stress', color: '#ef4444' };
}

// Update analysis status with different types
function updateAnalysisStatus(message, type = 'info') {
    const statusElement = document.getElementById('analysisStatus');
    if (!statusElement) return;
    
    const icons = {
        info: 'fa-info-circle',
        success: 'fa-check-circle',
        warning: 'fa-exclamation-triangle',
        error: 'fa-exclamation-triangle'
    };
    
    const colors = {
        info: 'bg-blue-50 text-blue-700',
        success: 'bg-green-50 text-green-700',
        warning: 'bg-yellow-50 text-yellow-700',
        error: 'bg-red-50 text-red-700'
    };
    
    statusElement.className = `mt-4 p-4 rounded-lg ${colors[type]}`;
    statusElement.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${icons[type]} mr-2"></i>
            <span>${message}</span>
        </div>
    `;
}

// Show personalized recommendations
async function showRecommendations() {
    try {
        const response = await fetch('/get_wellness_recommendations');
        const recommendations = await response.json();
        
        const panel = document.getElementById('recommendationsPanel');
        const content = document.getElementById('recommendationsContent');
        
        if (!panel || !content) return;
        
        let html = `
            <div class="bg-blue-50 p-4 rounded-lg mb-4 fade-in">
                <p class="text-blue-800 font-medium">${recommendations.encouragement}</p>
            </div>`;
        
        if (recommendations.immediate_techniques && recommendations.immediate_techniques.length > 0) {
            html += '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">';
            
            recommendations.immediate_techniques.forEach(technique => {
                html += `
                    <div class="technique-card fade-in">
                        <h4 class="font-semibold text-gray-800 mb-2">${technique.name}</h4>
                        <p class="text-gray-600 text-sm mb-3">${technique.description}</p>
                        <p class="text-xs text-gray-500 mb-2">Duration: ${technique.duration}</p>
                        <button class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600 transition-colors" 
                                onclick="startTechnique('${technique.id}')">
                            <i class="fas fa-play mr-1"></i>Start Exercise
                        </button>
                    </div>`;
            });
            
            html += '</div>';
        }
        
        content.innerHTML = html;
        panel.style.display = 'block';
        panel.classList.add('fade-in');
        
    } catch (err) {
        console.error('Error loading recommendations:', err);
    }
}

// Start a wellness technique
function startTechnique(techniqueId) {
    // This could open a modal with guided instructions
    console.log(`Starting technique: ${techniqueId}`);
    
    // For now, show a simple alert
    const message = `Starting wellness technique. Follow the guided instructions.`;
    showNotification(message, 'info');
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    const colors = {
        info: 'bg-blue-500',
        success: 'bg-green-500',
        warning: 'bg-yellow-500',
        error: 'bg-red-500'
    };
    
    notification.className = `fixed top-4 right-4 ${colors[type]} text-white p-4 rounded-lg shadow-lg z-50 fade-in`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Update trend visualization (mini chart in overlay)
function updateTrendVisualization(trendData) {
    // This could update a small chart showing recent stress trends
    // For now, we'll just log the data
    console.log('Stress trend data:', trendData);
}

// Event listeners
if (startBtn) startBtn.addEventListener('click', startMonitoring);
if (stopBtn) stopBtn.addEventListener('click', stopMonitoring);

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Check if browser supports getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateAnalysisStatus(
            'Your browser does not support webcam access. Please use a modern browser.',
            'error'
        );
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === ' ' && e.ctrlKey) { // Ctrl+Space
            e.preventDefault();
            if (isMonitoring) {
                stopMonitoring();
            } else {
                startMonitoring();
            }
        }
    });
});

// Handle page visibility changes (pause when tab not active)
document.addEventListener('visibilitychange', function() {
    if (document.hidden && isMonitoring) {
        // Optionally pause monitoring when tab is not visible
        console.log('Tab hidden - monitoring continues in background');
    } else if (!document.hidden && isMonitoring) {
        console.log('Tab visible - monitoring active');
    }
});

// Export functions for use in other scripts
window.aiWellnessApp = {
    startMonitoring,
    stopMonitoring,
    updateStressDisplay,
    showRecommendations,
    showNotification
};