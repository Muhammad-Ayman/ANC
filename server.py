"""
Flask server for Adaptive Noise Cancellation web interface.
Exposes REST API endpoints to process audio with LMS, NLMS, and RLS algorithms.
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pathlib import Path
import os
import json
import wave
import struct
import math
from typing import List, Tuple
import tempfile
import shutil

# Import the noise cancellation functions
from adaptive_noise_cancellation import (
    lms_filter, nlms_filter, rls_filter,
    read_wav, write_wav, remove_dc, normalize,
    running_rms, rms
)

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs_web')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Test audio files directories
TEST_AUDIO_DIRS = [Path('aud'), Path('aud2')]


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/test-files', methods=['GET'])
def get_test_files():
    """List all available test audio files."""
    files = []
    for dir_path in TEST_AUDIO_DIRS:
        if dir_path.exists():
            for file_path in dir_path.glob('*.wav'):
                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'dir': dir_path.name,
                    'size': file_path.stat().st_size
                })
    return jsonify({'files': files})


@app.route('/api/audio/<path:filepath>', methods=['GET'])
def get_audio(filepath):
    """Serve audio file for playback."""
    file_path = Path(filepath)
    if file_path.exists() and file_path.suffix == '.wav':
        return send_file(file_path, mimetype='audio/wav')
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle audio file uploads."""
    try:
        if 'noisy' not in request.files or 'noise' not in request.files:
            return jsonify({'error': 'Both noisy and noise files required'}), 400
        
        noisy_file = request.files['noisy']
        noise_file = request.files['noise']
        
        if noisy_file.filename == '' or noise_file.filename == '':
            return jsonify({'error': 'No selected files'}), 400
        
        # Save uploaded files
        noisy_path = UPLOAD_FOLDER / 'uploaded_noisy.wav'
        noise_path = UPLOAD_FOLDER / 'uploaded_noise.wav'
        
        noisy_file.save(noisy_path)
        noise_file.save(noise_path)
        
        # Get audio info
        rate_noisy, noisy_data = read_wav(noisy_path)
        rate_noise, noise_data = read_wav(noise_path)
        
        return jsonify({
            'success': True,
            'noisy': {
                'path': str(noisy_path),
                'rate': rate_noisy,
                'samples': len(noisy_data),
                'duration': len(noisy_data) / rate_noisy
            },
            'noise': {
                'path': str(noise_path),
                'rate': rate_noise,
                'samples': len(noise_data),
                'duration': len(noise_data) / rate_noise
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_audio():
    """Process audio with selected algorithm."""
    try:
        data = request.json
        
        # Get file paths
        noisy_path = Path(data['noisyPath'])
        noise_path = Path(data['noisePath'])
        
        if not noisy_path.exists() or not noise_path.exists():
            return jsonify({'error': 'Audio files not found'}), 404
        
        # Read audio files
        rate_noisy, noisy = read_wav(noisy_path)
        rate_noise, noise = read_wav(noise_path)
        
        if rate_noisy != rate_noise:
            return jsonify({'error': 'Sample rates must match'}), 400
        
        # Trim to same length
        length = min(len(noisy), len(noise))
        noisy = noisy[:length]
        noise = noise[:length]
        
        # Preprocess
        noisy = normalize(remove_dc(noisy))
        noise = normalize(remove_dc(noise))
        
        # Get algorithm and parameters
        algorithm = data.get('algorithm', 'lms')
        params = data.get('parameters', {})
        
        # Process based on algorithm
        if algorithm == 'lms':
            order = params.get('order', 12)
            mu = params.get('mu', 0.025)
            clean = lms_filter(noise, noisy, order=order, mu=mu)
            algo_params = {'order': order, 'mu': mu}
            
        elif algorithm == 'nlms':
            order = params.get('order', 12)
            mu = params.get('mu', 0.5)
            eps = params.get('eps', 1e-6)
            clean = nlms_filter(noise, noisy, order=order, mu=mu, eps=eps)
            algo_params = {'order': order, 'mu': mu, 'eps': eps}
            
        elif algorithm == 'rls':
            order = params.get('order', 15)
            lam = params.get('lambda', 0.949)
            delta = params.get('delta', 0.06)
            clean = rls_filter(noise, noisy, order=order, lam=lam, delta=delta)
            algo_params = {'order': order, 'lambda': lam, 'delta': delta}
            
        else:
            return jsonify({'error': f'Unknown algorithm: {algorithm}'}), 400
        
        # Save output
        output_filename = f'output_{algorithm}.wav'
        output_path = OUTPUT_FOLDER / output_filename
        write_wav(output_path, rate_noisy, clean)
        
        # Calculate convergence curve
        window = max(1, int(rate_noisy * 0.05))  # 50ms window
        convergence = running_rms(clean, window)
        
        # Downsample convergence data for plotting (every 100 samples)
        step = max(1, len(convergence) // 1000)
        convergence_data = [
            {
                'time': i / rate_noisy,
                'value': convergence[i]
            }
            for i in range(0, len(convergence), step)
        ]
        
        # Calculate metrics
        input_rms = rms(noisy)
        output_rms = rms(clean)
        
        return jsonify({
            'success': True,
            'output': {
                'path': str(output_path),
                'filename': output_filename,
                'samples': len(clean),
                'duration': len(clean) / rate_noisy,
                'rate': rate_noisy
            },
            'metrics': {
                'inputRMS': input_rms,
                'outputRMS': output_rms,
                'reduction': (1 - output_rms / input_rms) * 100 if input_rms > 0 else 0
            },
            'convergence': convergence_data,
            'algorithm': algorithm,
            'parameters': algo_params
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/waveform/<path:filepath>', methods=['GET'])
def get_waveform(filepath):
    """Get waveform data for visualization."""
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        rate, samples = read_wav(file_path)
        
        # Downsample for visualization (max 2000 points)
        step = max(1, len(samples) // 2000)
        waveform = [
            {
                'time': i / rate,
                'amplitude': samples[i]
            }
            for i in range(0, len(samples), step)
        ]
        
        return jsonify({
            'waveform': waveform,
            'rate': rate,
            'samples': len(samples),
            'duration': len(samples) / rate
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed audio file."""
    file_path = OUTPUT_FOLDER / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True, download_name=filename)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("🚀 Starting Adaptive Noise Cancellation Server...")
    print("📁 Uploads folder:", UPLOAD_FOLDER.absolute())
    print("📁 Outputs folder:", OUTPUT_FOLDER.absolute())
    print("🌐 Server running at: http://localhost:5000")
    print("\n✨ Open your browser to http://localhost:5000\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
