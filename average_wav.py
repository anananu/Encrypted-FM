import numpy as np
import scipy.io.wavfile as wavfile
import sys

def average_wav_files(file_list):
    if len(file_list) < 2:
        raise ValueError("You must specify at least two WAV files.")

    sample_rate, data = wavfile.read(file_list[0])
    num_channels = data.shape[1] if len(data.shape) > 1 else 1
    min_samples = len(data)
    
    total_data = np.zeros((min_samples, num_channels), dtype=np.float64)
    
    for file in file_list:
        sr, d = wavfile.read(file)
        if sr != sample_rate:
            raise ValueError("All WAV files must have the same sample rate.")
        num_samples = len(d)
        min_samples = min(min_samples, num_samples)
        d = d[:min_samples]  

        if len(d.shape) > 1:
            if d.shape[1] != num_channels:
                raise ValueError("All WAV files must have the same number of channels.")
            total_data[:min_samples] += d.astype(np.float64)
        else:
            if num_channels != 1:
                raise ValueError("WAV files must have the same number of channels.")
            d_stereo = np.stack([d, d], axis=1)
            total_data[:min_samples] += d_stereo.astype(np.float64)
    
    average_data = total_data / len(file_list)
    
    average_data = np.clip(average_data, -32768, 32767)
    average_data = average_data.astype(np.int16)
    
    return sample_rate, average_data

def main():
    if len(sys.argv) < 3:
        print("python script.py file1.wav file2.wav [file3.wav ...]")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    
    sample_rate, averaged_data = average_wav_files(input_files)
    
    output_filename = 'output_averaged.wav'
    wavfile.write(output_filename, sample_rate, averaged_data)
    print(f"The resulting WAV file was saved as'{output_filename}'.")

if __name__ == "__main__":
    main()
