import numpy as np
import soundfile as sf

# Load noise and audio
noise, sr_noise = sf.read("aud/audio_noise.wav")
audio, sr_audio = sf.read("aud2/clean.wav")

print(f"Noise: {sr_noise} Hz, {len(noise)} samples = {len(noise)/sr_noise:.2f} sec")
print(f"Audio: {sr_audio} Hz, {len(audio)} samples = {len(audio)/sr_audio:.2f} sec")

# Simple resampling using linear interpolation if sample rates differ
if sr_audio != sr_noise:
    print(f"\n⚠️  Sample rates don't match!")
    print(f"Resampling audio from {sr_audio} Hz to {sr_noise} Hz...")
    
    # Calculate new number of samples
    duration = len(audio) / sr_audio  # duration in seconds
    new_length = int(duration * sr_noise)
    
    # Linear interpolation resampling
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    audio = np.interp(new_indices, old_indices, audio)
    
    print(f"After resampling: {len(audio)} samples = {len(audio)/sr_noise:.2f} sec")

# Trim to same length
min_len = min(len(audio), len(noise))
clean = audio[:min_len]
noise_trimmed = noise[:min_len]

print(f"\n📊 Final output: {min_len} samples at {sr_noise} Hz = {min_len/sr_noise:.2f} seconds")

# Scale noise for desired SNR
snr_db = 5
alpha = np.sqrt(np.mean(clean**2) / (np.mean(noise_trimmed**2) * (10**(snr_db/10))))
noisy = clean + alpha * noise_trimmed

print(f"SNR: {snr_db} dB (noise multiplier: {alpha:.3f})")

# Save noisy mixture at noise sample rate
sf.write("noisy_sample.wav", noisy, sr_noise)
print(f"\n✅ Saved 'noisy_sample.wav' at {sr_noise} Hz!")
