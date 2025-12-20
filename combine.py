import numpy as np
import soundfile as sf

# Load noise
noise, sr_noise = sf.read("aud/audio_noise.wav")
audio, sr = sf.read("aud/audio.wav")
print(sr_noise, sr)
# Trim to same length
min_len = min(len(audio), len(noise))
clean = audio[:min_len]
noise = noise[:min_len]

# Scale noise for desired SNR (e.g., 5 dB)
snr_db = 5
alpha = np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * (10**(snr_db/10))))
noisy = clean + alpha * noise

# Save noisy mixture
sf.write("noisy_sample.wav", noisy, sr_noise)
print("Noisy mixture saved!")
