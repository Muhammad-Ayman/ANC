/**
 * Main Application Logic
 * Handles UI interactions, file uploads, API communication, and orchestrates visualization
 */

// Global state
const state = {
    noisyPath: null,
    noisePath: null,
    testFiles: [],
    noisyWaveform: null,
    noiseWaveform: null,
    outputWaveform: null,
    visualizer: null
};

// API Base URL
const API_BASE = window.location.origin;

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    state.visualizer = new AudioVisualizer();
    initializeUI();
    loadTestFiles();
});

/**
 * Initialize UI event listeners
 */
function initializeUI() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tab = e.target.dataset.tab;
            switchTab(tab);
        });
    });

    // Upload boxes
    setupUploadBox('noisy');
    setupUploadBox('noise');

    // Test samples
    document.getElementById('test-select').addEventListener('change', (e) => {
        document.getElementById('load-test-btn').disabled = !e.target.value;
    });

    document.getElementById('load-test-btn').addEventListener('click', loadTestSample);

    // Algorithm selection
    document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            switchAlgorithm(e.target.value);
        });
    });

    // Process button
    document.getElementById('process-btn').addEventListener('click', processAudio);

    // Download button
    document.getElementById('download-btn').addEventListener('click', downloadOutput);
}

/**
 * Switch between tabs
 */
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tab}-tab`);
    });
}

/**
 * Setup upload box for drag & drop and click
 */
function setupUploadBox(type) {
    const box = document.getElementById(`${type}-upload`);
    const input = document.getElementById(`${type}-file`);
    const info = document.getElementById(`${type}-info`);

    // Click to upload
    box.addEventListener('click', () => input.click());

    // File input change
    input.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            await handleFileUpload(type, file, info, box);
        }
    });

    // Drag & drop
    box.addEventListener('dragover', (e) => {
        e.preventDefault();
        box.style.borderColor = '#6366f1';
    });

    box.addEventListener('dragleave', () => {
        box.style.borderColor = '';
    });

    box.addEventListener('drop', async (e) => {
        e.preventDefault();
        box.style.borderColor = '';

        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.wav')) {
            input.files = e.dataTransfer.files;
            await handleFileUpload(type, file, info, box);
        } else {
            showNotification('Please drop a WAV file', 'error');
        }
    });
}

/**
 * Handle file upload
 */
async function handleFileUpload(type, file, infoEl, boxEl) {
    infoEl.textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    infoEl.classList.add('active');
    boxEl.classList.add('active');

    // Check if both files are uploaded
    const noisyFile = document.getElementById('noisy-file').files[0];
    const noiseFile = document.getElementById('noise-file').files[0];

    if (noisyFile && noiseFile) {
        await uploadFiles(noisyFile, noiseFile);
    }
}

/**
 * Upload files to server
 */
async function uploadFiles(noisyFile, noiseFile) {
    try {
        showNotification('Uploading files...', 'info');

        const formData = new FormData();
        formData.append('noisy', noisyFile);
        formData.append('noise', noiseFile);

        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            state.noisyPath = data.noisy.path;
            state.noisePath = data.noise.path;

            showNotification('Files uploaded successfully!', 'success');
            await loadAudioPreview();
        } else {
            showNotification(data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Upload failed: ' + error.message, 'error');
    }
}

/**
 * Load test files from server
 */
async function loadTestFiles() {
    try {
        const response = await fetch(`${API_BASE}/api/test-files`);
        const data = await response.json();

        state.testFiles = data.files;

        const select = document.getElementById('test-select');

        // Group files by directory
        const grouped = {};
        data.files.forEach(file => {
            if (!grouped[file.dir]) grouped[file.dir] = [];
            grouped[file.dir].push(file);
        });

        // Populate select
        Object.entries(grouped).forEach(([dir, files]) => {
            const optgroup = document.createElement('optgroup');
            optgroup.label = dir;

            files.forEach(file => {
                const option = document.createElement('option');
                option.value = file.path;
                option.textContent = file.name;
                optgroup.appendChild(option);
            });

            select.appendChild(optgroup);
        });
    } catch (error) {
        console.error('Failed to load test files:', error);
    }
}

/**
 * Load selected test sample
 */
async function loadTestSample() {
    const select = document.getElementById('test-select');
    const selectedPath = select.value;

    if (!selectedPath) return;

    // For test samples, we need both noisy and noise reference
    // Assume we're selecting from aud2 which has noisy_sample.wav
    // and we'll use audio_noise.wav from aud as reference

    const fileName = selectedPath.split(/[/\\]/).pop();

    if (fileName === 'noisy_sample.wav') {
        state.noisyPath = selectedPath;
        state.noisePath = 'aud/audio_noise.wav';
        showNotification('Test sample loaded!', 'success');
        await loadAudioPreview();
    } else if (fileName === 'audio.wav') {
        state.noisyPath = selectedPath;
        state.noisePath = 'aud/audio_noise.wav';
        showNotification('Test sample loaded!', 'success');
        await loadAudioPreview();
    } else {
        showNotification('Please select a noisy sample file', 'info');
    }
}

/**
 * Load audio preview with waveforms
 */
async function loadAudioPreview() {
    try {
        // Show preview section
        document.getElementById('preview-section').style.display = 'block';
        document.getElementById('algorithm-section').style.display = 'block';

        // Load audio players
        const noisyPlayer = document.getElementById('noisy-player');
        const noisePlayer = document.getElementById('noise-player');

        document.getElementById('noisy-source').src = `${API_BASE}/api/audio/${state.noisyPath}`;
        document.getElementById('noise-source').src = `${API_BASE}/api/audio/${state.noisePath}`;

        noisyPlayer.load();
        noisePlayer.load();

        // Load waveforms
        await loadWaveform(state.noisyPath, 'noisy-canvas');
        await loadWaveform(state.noisePath, 'noise-canvas');

        // Smooth scroll to preview
        document.getElementById('preview-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (error) {
        console.error('Failed to load preview:', error);
        showNotification('Failed to load audio preview', 'error');
    }
}

/**
 * Load and visualize waveform
 */
async function loadWaveform(filepath, canvasId) {
    try {
        const response = await fetch(`${API_BASE}/api/waveform/${filepath}`);
        const data = await response.json();

        if (data.waveform) {
            const canvas = document.getElementById(canvasId);
            state.visualizer.initCanvas(canvas);
            state.visualizer.drawWaveform(canvas, data.waveform);

            // Store waveform data
            if (canvasId === 'noisy-canvas') {
                state.noisyWaveform = data.waveform;
            } else if (canvasId === 'noise-canvas') {
                state.noiseWaveform = data.waveform;
            }
        }
    } catch (error) {
        console.error(`Failed to load waveform for ${filepath}:`, error);
    }
}

/**
 * Switch algorithm and show corresponding parameters
 */
function switchAlgorithm(algo) {
    document.querySelectorAll('.param-group').forEach(group => {
        group.classList.remove('active');
    });
    document.getElementById(`${algo}-params`).classList.add('active');
}

/**
 * Process audio with selected algorithm
 */
async function processAudio() {
    try {
        if (!state.noisyPath || !state.noisePath) {
            showNotification('Please upload or select audio files first', 'error');
            return;
        }

        // Get selected algorithm
        const algorithm = document.querySelector('input[name="algorithm"]:checked').value;

        // Get parameters
        const parameters = getAlgorithmParameters(algorithm);

        // Show processing overlay
        document.getElementById('processing-overlay').style.display = 'flex';
        document.getElementById('processing-status').textContent = `Applying ${algorithm.toUpperCase()} filter...`;

        // Send processing request
        const response = await fetch(`${API_BASE}/api/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                noisyPath: state.noisyPath,
                noisePath: state.noisePath,
                algorithm: algorithm,
                parameters: parameters
            })
        });

        const data = await response.json();

        // Hide processing overlay
        document.getElementById('processing-overlay').style.display = 'none';

        if (data.success) {
            showNotification('Processing complete!', 'success');
            displayResults(data);
        } else {
            showNotification(data.error || 'Processing failed', 'error');
        }
    } catch (error) {
        document.getElementById('processing-overlay').style.display = 'none';
        console.error('Processing error:', error);
        showNotification('Processing failed: ' + error.message, 'error');
    }
}

/**
 * Get algorithm parameters from UI
 */
function getAlgorithmParameters(algo) {
    const params = {};

    if (algo === 'lms') {
        params.order = parseInt(document.getElementById('lms-order').value);
        params.mu = parseFloat(document.getElementById('lms-mu').value);
    } else if (algo === 'rls') {
        params.order = parseInt(document.getElementById('rls-order').value);
        params.lambda = parseFloat(document.getElementById('rls-lambda').value);
        params.delta = parseFloat(document.getElementById('rls-delta').value);
    }

    return params;
}

/**
 * Display processing results
 */
async function displayResults(data) {
    // Show results section
    document.getElementById('results-section').style.display = 'block';

    // Load output audio
    const outputPlayer = document.getElementById('output-player');
    document.getElementById('output-source').src = `${API_BASE}/api/audio/${data.output.path}`;
    outputPlayer.load();

    // Load output waveform
    await loadOutputWaveform(data.output.path);

    // Display metrics
    document.getElementById('metric-algo').textContent = data.algorithm.toUpperCase();
    document.getElementById('metric-input-rms').textContent = data.metrics.inputRMS.toFixed(6);
    document.getElementById('metric-output-rms').textContent = data.metrics.outputRMS.toFixed(6);
    document.getElementById('metric-reduction').textContent = data.metrics.reduction.toFixed(2) + '%';

    // Display parameters used
    const paramsDiv = document.getElementById('params-used');
    paramsDiv.innerHTML = '<strong>Parameters:</strong><br>' +
        Object.entries(data.parameters)
            .map(([key, val]) => `${key}: ${typeof val === 'number' ? val.toFixed(6) : val}`)
            .join(', ');

    // Draw convergence plot
    const convergenceCanvas = document.getElementById('convergence-canvas');
    state.visualizer.initCanvas(convergenceCanvas);
    state.visualizer.drawConvergence(convergenceCanvas, data.convergence);

    // Draw comparison
    if (state.noisyWaveform && state.outputWaveform) {
        const comparisonCanvas = document.getElementById('comparison-canvas');
        state.visualizer.initCanvas(comparisonCanvas);
        state.visualizer.drawComparison(comparisonCanvas, state.noisyWaveform, state.outputWaveform);
    }

    // Store output path for download
    state.outputFilename = data.output.filename;

    // Scroll to results
    setTimeout(() => {
        document.getElementById('results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

/**
 * Load output waveform
 */
async function loadOutputWaveform(filepath) {
    try {
        const response = await fetch(`${API_BASE}/api/waveform/${filepath}`);
        const data = await response.json();

        if (data.waveform) {
            const canvas = document.getElementById('output-canvas');
            state.visualizer.initCanvas(canvas);
            state.visualizer.drawWaveform(canvas, data.waveform, '#10b981');
            state.outputWaveform = data.waveform;
        }
    } catch (error) {
        console.error('Failed to load output waveform:', error);
    }
}

/**
 * Download processed audio
 */
function downloadOutput() {
    if (state.outputFilename) {
        window.location.href = `${API_BASE}/api/download/${state.outputFilename}`;
    }
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#6366f1'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        z-index: 10000;
        font-weight: 500;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(notification);

    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
