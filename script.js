// Configuration
const API_URL = 'http://localhost:5000/api';

// State
let stats = {
    totalScans: 0,
    threatsDetected: 0,
    startTime: Date.now()
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeMatrixBackground();
    checkSystemHealth();
    setupEventListeners();
    startUptime();
});

// Matrix Background Animation
function initializeMatrixBackground() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const matrixBg = document.getElementById('matrixBg');
    
    matrixBg.appendChild(canvas);
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
    const fontSize = 14;
    const columns = canvas.width / fontSize;
    const drops = Array(Math.floor(columns)).fill(1);
    
    function drawMatrix() {
        ctx.fillStyle = 'rgba(5, 8, 20, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#00ff41';
        ctx.font = `${fontSize}px monospace`;
        
        for (let i = 0; i < drops.length; i++) {
            const text = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
            
            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
    }
    
    setInterval(drawMatrix, 50);
    
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

// Event Listeners
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // File upload
    fileInput.addEventListener('change', handleFileUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Manual analysis
    analyzeBtn.addEventListener('click', handleManualAnalysis);
}

// System Health Check
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            updateSystemStatus('SYSTEM ONLINE', true);
        } else {
            updateSystemStatus('MODEL NOT LOADED', false);
        }
    } catch (error) {
        updateSystemStatus('API CONNECTION FAILED', false);
        console.error('Health check failed:', error);
    }
}

function updateSystemStatus(message, isHealthy) {
    const statusText = document.getElementById('statusText');
    const statusDot = document.querySelector('.status-dot');
    
    statusText.textContent = message;
    
    if (isHealthy) {
        statusDot.style.background = 'var(--primary-green)';
    } else {
        statusDot.style.background = 'var(--red-alert)';
    }
}

// File Handling
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    // Check if it's an image
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (PNG, JPG, JPEG)');
        return;
    }
    
    try {
        // Convert image to base64
        const base64Image = await fileToBase64(file);
        
        // Show preview
        const previewImage = document.getElementById('previewImage');
        if (previewImage) {
            previewImage.src = base64Image;
        }
        
        // Analyze the image
        analyzeImage(base64Image);
    } catch (error) {
        alert('Error reading file: ' + error.message);
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Manual Analysis
function handleManualAnalysis() {
    const manualInput = document.getElementById('manualFeatures').value.trim();
    
    if (!manualInput) {
        alert('Please enter base64 image data');
        return;
    }
    
    try {
        // Validate it's a base64 image string
        if (!manualInput.startsWith('data:image/') && !manualInput.match(/^[A-Za-z0-9+/=]+$/)) {
            throw new Error('Invalid base64 image format');
        }
        
        // Show preview if possible
        const previewImage = document.getElementById('previewImage');
        if (previewImage) {
            const imageData = manualInput.startsWith('data:') ? manualInput : 'data:image/png;base64,' + manualInput;
            previewImage.src = imageData;
        }
        
        analyzeImage(manualInput);
    } catch (error) {
        alert('Invalid input: ' + error.message);
    }
}

// Analysis
async function analyzeImage(imageData) {
    updateScanIndicator(true);
    
    try {
        const response = await fetch(`${API_URL}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                timestamp: new Date().toISOString()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
        updateStats(result);
        
    } catch (error) {
        alert('Analysis failed: ' + error.message);
        console.error('Analysis error:', error);
    } finally {
        updateScanIndicator(false);
    }
}

function updateScanIndicator(scanning) {
    const indicator = document.getElementById('scanIndicator');
    if (scanning) {
        indicator.innerHTML = '<span class="pulse"></span>SCANNING...';
    } else {
        indicator.innerHTML = '<span class="pulse"></span>READY';
    }
}

// Display Results
function displayResults(result) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Update timestamp
    const timestamp = document.getElementById('timestamp');
    timestamp.textContent = new Date().toLocaleString();
    
    // Update prediction
    const predictionText = document.getElementById('predictionText');
    predictionText.textContent = result.prediction;
    
    // Update threat icon
    const threatIcon = document.getElementById('threatIcon');
    const iconMap = {
        'Benign': '✓',
        'Virus': '☠',
        'Worm': '🐛',
        'Trojan': '⚠',
        'Spyware': '👁',
        'Adware': '⚡',
        'Downloader': '⬇',
        'Backdoor': '🚪'
    };
    threatIcon.textContent = iconMap[result.prediction] || '⚠';
    
    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercent = (result.confidence * 100).toFixed(2);
    
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceFill.style.width = `${confidencePercent}%`;
    
    // Update threat level
    const threatLabel = document.getElementById('threatLabel');
    const threatMeter = document.getElementById('threatMeter');
    
    threatLabel.textContent = result.threat_level;
    threatLabel.className = `threat-label threat-${result.threat_color}`;
    
    const levelMap = { 'Safe': 25, 'Low': 50, 'High': 75, 'Critical': 100 };
    threatMeter.style.width = `${levelMap[result.threat_level]}%`;
    threatMeter.className = `meter-fill bg-${result.threat_color}`;
    
    // Update probability chart
    displayProbabilityChart(result.probabilities);
    
    // Update recommendations
    displayRecommendations(result);
}

function displayProbabilityChart(probabilities) {
    const chart = document.getElementById('probabilityChart');
    chart.innerHTML = '';
    
    // Sort by probability
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sorted.forEach(([className, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        
        const percent = (prob * 100).toFixed(2);
        
        item.innerHTML = `
            <div class="prob-label">${className}</div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: ${percent}%"></div>
            </div>
            <div class="prob-value">${percent}%</div>
        `;
        
        chart.appendChild(item);
    });
}

function displayRecommendations(result) {
    const recommendations = document.getElementById('recommendations');
    recommendations.innerHTML = '';
    
    const recMap = {
        'Benign': [
            { icon: '✓', text: 'File appears safe. No immediate action required.' },
            { icon: '📊', text: 'Continue regular monitoring and security practices.' }
        ],
        'Virus': [
            { icon: '☠', text: 'CRITICAL: Quarantine the file immediately and run full system scan.' },
            { icon: '🔍', text: 'Check for file replication and system modifications.' },
            { icon: '💾', text: 'Restore affected files from clean backups.' }
        ],
        'Worm': [
            { icon: '🐛', text: 'ALERT: Isolate infected system from network immediately.' },
            { icon: '🔒', text: 'Block network propagation and scan all connected devices.' },
            { icon: '📡', text: 'Monitor network traffic for suspicious activity.' }
        ],
        'Trojan': [
            { icon: '⚠', text: 'ALERT: Quarantine the file immediately.' },
            { icon: '🔍', text: 'Perform a full system scan for additional threats.' },
            { icon: '🔒', text: 'Review and update system access controls.' }
        ],
        'Spyware': [
            { icon: '👁', text: 'WARNING: Device may be under surveillance.' },
            { icon: '🔐', text: 'Change all passwords from a secure device.' },
            { icon: '📱', text: 'Review data access permissions and network activity.' }
        ],
        'Adware': [
            { icon: '⚡', text: 'Remove adware using anti-malware tools.' },
            { icon: '🌐', text: 'Check browser extensions and installed programs.' },
            { icon: '🔧', text: 'Reset browser settings to defaults.' }
        ],
        'Downloader': [
            { icon: '⬇', text: 'WARNING: File may download additional malware.' },
            { icon: '🚫', text: 'Block outbound connections and quarantine immediately.' },
            { icon: '🔍', text: 'Scan for recently downloaded or modified files.' }
        ],
        'Backdoor': [
            { icon: '🚪', text: 'CRITICAL: System may be remotely accessible to attackers.' },
            { icon: '🔒', text: 'Disconnect from network and change all credentials.' },
            { icon: '📞', text: 'Contact security team immediately for incident response.' }
        ]
    };
    
    const recs = recMap[result.prediction] || recMap['Trojan'];
    
    recs.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.innerHTML = `
            <div class="recommendation-icon">${rec.icon}</div>
            <div class="recommendation-text">${rec.text}</div>
        `;
        recommendations.appendChild(item);
    });
}

// Stats
function updateStats(result) {
    stats.totalScans++;
    
    if (result.prediction !== 'Benign') {
        stats.threatsDetected++;
    }
    
    document.getElementById('totalScans').textContent = stats.totalScans;
    document.getElementById('threatsDetected').textContent = stats.threatsDetected;
}

function startUptime() {
    setInterval(() => {
        const elapsed = Date.now() - stats.startTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);
        
        const uptime = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        document.getElementById('uptime').textContent = uptime;
    }, 1000);
}

// Export for console testing
window.malwareAnalyzer = {
    analyzeImage,
    checkSystemHealth
};

console.log('Malware Analyzer initialized. Use window.malwareAnalyzer for testing.');
console.log('Note: Upload an image file to test the system.');