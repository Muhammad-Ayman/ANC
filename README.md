# Adaptive Noise Cancellation Using LMS and RLS Filters

A comprehensive implementation and comparative study of adaptive noise cancellation techniques using Least Mean Square (LMS) and Recursive Least Squares (RLS) adaptive filters.

## Project Overview

This project implements adaptive noise cancellation algorithms to remove unwanted noise from speech signals. Two implementations are provided:

1. **Custom Implementation** (`adaptive_noise_cancellation.py`): Pure Python implementation using only standard library
2. **Padasip Implementation** (`adaptive_noise_cancellation_padasip.py`): Implementation using the padasip library

## Features

- LMS (Least Mean Square) adaptive filter implementation
- RLS (Recursive Least Squares) adaptive filter implementation
- Grid search for optimal parameter selection
- Convergence rate analysis
- Signal visualization and comparison
- Comprehensive LaTeX report

## Requirements

### For Custom Implementation
- Python 3.7+
- matplotlib (optional, for visualization)
- Standard library only (wave, struct, math, etc.)

### For Padasip Implementation
- Python 3.7+
- numpy
- padasip
- matplotlib (optional, for visualization)

## Installation

```bash
# Install dependencies for padasip implementation
pip install padasip numpy matplotlib
```

## Usage

### Custom Implementation

```bash
# Run with default parameters
python adaptive_noise_cancellation.py

# Run with grid search for optimal parameters
python adaptive_noise_cancellation.py --grid

# Specify custom parameters
python adaptive_noise_cancellation.py \
    --lms-order 32 --lms-mu 0.01 \
    --rls-order 16 --rls-lam 0.995 --rls-delta 0.01

# Process different audio files
python adaptive_noise_cancellation.py \
    --noisy aud/audio2.wav \
    --noise aud/audio2_noise.wav
```

### Padasip Implementation

```bash
# Run both algorithms
python adaptive_noise_cancellation_padasip.py --algo both

# Run only LMS
python adaptive_noise_cancellation_padasip.py --algo lms \
    --lms-order 32 --lms-mu 0.01

# Run only RLS
python adaptive_noise_cancellation_padasip.py --algo rls \
    --rls-order 16 --rls-lam 0.995 --rls-delta 0.01
```

## Project Structure

```
ANC/
├── aud/                          # Input audio files
│   ├── audio.wav                 # Noisy speech
│   ├── audio_noise.wav           # Noise reference
│   ├── audio2.wav
│   └── audio2_noise.wav
├── outputs_basic/                # Custom implementation outputs
│   ├── audio_lms.wav
│   ├── audio_rls.wav
│   ├── signals.png
│   ├── convergence.png
│   └── convergence.csv
├── outputs_padasip/              # Padasip implementation outputs
│   ├── audio_lms_pada.wav
│   ├── audio_rls_pada.wav
│   ├── signals_pada.png
│   ├── convergence_pada.png
│   └── convergence_pada.csv
├── adaptive_noise_cancellation.py
├── adaptive_noise_cancellation_padasip.py
├── report.tex                    # LaTeX report
└── README.md
```

## Parameters

### LMS Parameters
- **Order**: Filter order (typically 8-64, recommended: 16-32)
- **μ (mu)**: Step size (typically 0.001-0.1, recommended: 0.005-0.01)

### RLS Parameters
- **Order**: Filter order (typically 4-32, recommended: 8-16)
- **λ (lambda)**: Forgetting factor (typically 0.9-0.999, recommended: 0.99-0.995)
- **δ (delta)**: Initialization parameter (typically 0.001-0.1, recommended: 0.01-0.1)

## Results

The project includes a comprehensive LaTeX report (`report.tex`) covering:
- Methodology and algorithm descriptions
- Implementation details
- Experimental results and analysis
- Convergence rate comparisons
- Performance metrics

To compile the report:
```bash
pdflatex report.tex
```


## Supervisor

Dr. Amr Wagih

## License

This project is for educational purposes.

