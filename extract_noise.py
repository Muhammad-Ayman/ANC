# extract_noise.py
import soundfile as sf
import numpy as np
import argparse
import os

def extract_noise(clean_path, noisy_path, output_path="estimated_noise.wav"):
    # Read both files
    clean, sr1 = sf.read(clean_path)
    noisy, sr2 = sf.read(noisy_path)
    
    # Basic safety checks
    if sr1 != sr2:
        raise ValueError(f"Sample rates don't match: {sr1} vs {sr2}")
    if clean.shape != noisy.shape:
        min_len = min(len(clean), len(noisy))
        print(f"Length mismatch → truncating to {min_len} samples")
        clean = clean[:min_len]
        noisy = noisy[:min_len]
    
    # Convert to float if needed
    if clean.dtype != np.float32 and clean.dtype != np.float64:
        clean = clean.astype(np.float32) / np.iinfo(clean.dtype).max
    if noisy.dtype != np.float32 and noisy.dtype != np.float64:
        noisy = noisy.astype(np.float32) / np.iinfo(noisy.dtype).max
    
    # Subtract: noise = noisy - clean
    noise = noisy - clean
    
    # Optional: remove any DC offset that might appear due to tiny mismatches
    noise = noise - np.mean(noise)
    
    # Save the extracted noise
    sf.write(output_path, noise, sr1)
    print(f"Noise extracted and saved to: {output_path}")
    
    # Also save a quick check file (optional)
    sf.write("check_resynthesis.wav", clean + noise, sr1)
    print("Verification file saved: check_resynthesis.wav (should sound identical to noisy.wav)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract noise by subtracting clean speech from noisy speech")
    parser.add_argument("--clean", required=True, help="Path to clean speech WAV file")
    parser.add_argument("--noisy", required=True, help="Path to noisy speech WAV file")
    parser.add_argument("--output", default="estimated_noise.wav", help="Output noise file (default: estimated_noise.wav)")
    
    args = parser.parse_args()
    
    extract_noise(args.clean, args.noisy, args.output)