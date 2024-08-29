import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def load_wav(file_path):
    rate, data = wavfile.read(file_path)
    return rate, data

def save_wav(file_path, rate, data):
    wavfile.write(file_path, rate, data)

def high_pass_filter(audio_data, rate, cutoff=1000):
    # Apply a high-pass filter to remove lower frequencies
    nyquist = 0.5 * rate
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, audio_data)
    return filtered_data

def extract_noise(audio_data, rate):
    # High-pass filter to isolate noise
    noise_profile = high_pass_filter(audio_data, rate)
    return noise_profile

def normalize_audio(audio_data):
    # Normalize audio to ensure it's hearable
    max_val = np.max(np.abs(audio_data))
    normalized_data = (audio_data / max_val) * 32767
    return normalized_data.astype(np.int16)

def main():
    # Load your recorded song with noise
    rate, song_with_noise = load_wav(r"C:\Users\anana\OneDrive\Desktop\Recordings\MonsantoPark\LockedOutOfHeaven\LockedOutOfHeaven-BrunoMars.wav")
    
    # Extract noise profile by filtering the audio
    noise_profile = extract_noise(song_with_noise, rate)
    
    # Normalize the noise to make sure it's hearable
    normalized_noise = normalize_audio(noise_profile)
    
    # Save the noise profile as a separate audio file
    save_wav("extracted_noise.wav", rate, normalized_noise)
    print("Noise extracted and saved as extracted_noise.wav")

if __name__ == "__main__":
    main()
