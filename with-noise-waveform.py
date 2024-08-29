import wave
import binascii
import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def get_wav_header_info(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        header_info = {
            "Number of Channels": wf.getnchannels(),
            "Sample Width (bytes)": wf.getsampwidth(),
            "Frame Rate (samples/sec)": wf.getframerate(),
            "Number of Frames": wf.getnframes(),
            "Compression Type": wf.getcomptype(),
            "Compression Name": wf.getcompname()
        }
    return header_info

def print_wav_header_info(header_info, wav_file):
    print(f"File: {wav_file}")
    for key, value in header_info.items():
        print(f"{key}: {value}")
    print()

def read_wav_data(wav_file):
    with wave.open(wav_file, 'rb') as wf:
        raw_data = wf.readframes(wf.getnframes())
        return raw_data

def get_samples(raw_data, sample_width):
    num_samples = len(raw_data) // sample_width
    format_char = '<h' if sample_width == 2 else '<b' 
    
    samples = []
    for i in range(num_samples):
        sample = struct.unpack(format_char, raw_data[i*sample_width:(i+1)*sample_width])[0]
        samples.append(sample)
    
    return samples

def hex_to_normalized_decimal(raw_data, sample_width):
    num_samples = len(raw_data) // sample_width
    format_char = 'h' if sample_width == 2 else 'b'
    max_value = 2**(8*sample_width - 1)

    normalized_values = []
    for i in range(num_samples):
        sample = struct.unpack('<' + format_char, raw_data[i*sample_width:(i+1)*sample_width])[0]
        normalized_value = sample / max_value
        normalized_values.append(normalized_value)

    return normalized_values

def calculate_differences(normalized_values1, normalized_values2):
    num_samples = min(len(normalized_values1), len(normalized_values2))
    normalized_differences = []

    for i in range(num_samples):
        norm_value1 = normalized_values1[i]
        norm_value2 = normalized_values2[i]
        normalized_difference = norm_value1 - norm_value2
        normalized_differences.append(normalized_difference)

    return normalized_differences

def write_difference_wav(differences, output_wav_file, sample_rate, sample_width):
    if sample_width == 1:
        max_amplitude = 2**7 - 1  
    elif sample_width == 2:
        max_amplitude = 2**15 - 1 
    else:
        raise ValueError("Unsupported sample width. Only 1-byte and 2-byte samples are supported.")

    max_diff = max(differences)
    min_diff = min(differences)

    if max_diff > 1 or min_diff < -1:
        scale_factor = max_amplitude / max(abs(min_diff), abs(max_diff))
    else:
        scale_factor = max_amplitude

    print(f"Scale factor: {scale_factor}")

    with wave.open(output_wav_file, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        
        for diff in differences:
            
            scaled_diff = int(diff * scale_factor)
            scaled_diff = max(-max_amplitude, min(max_amplitude, scaled_diff))
            
            if sample_width == 1:
                wf.writeframes(struct.pack('<B', scaled_diff + 128)) 
            elif sample_width == 2:
                wf.writeframes(struct.pack('<h', scaled_diff))

def plot_histogram(differences):
    bins = np.arange(-2, 2, 0.1)
    hist, edges = np.histogram(differences, bins=bins)

    plt.hist(differences, bins=bins, edgecolor='black')
    plt.xlabel('Difference Intervals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Differences')
    plt.xticks(bins)
    plt.show()

    return hist, edges

def plot_noise_wave(differences):
    plt.plot(differences)
    plt.xlabel('Sample Index')
    plt.ylabel('Difference')
    plt.title('Noise Waveform')
    plt.show()

def compare_wav_files(file1, file2, output_type):
    header_info1 = get_wav_header_info(file1)
    header_info2 = get_wav_header_info(file2)

    print("Header Information for File 1:")
    print_wav_header_info(header_info1, file1)

    print("Header Information for File 2:")
    print_wav_header_info(header_info2, file2)

    raw_data1 = read_wav_data(file1)
    raw_data2 = read_wav_data(file2)

    sample_width1 = header_info1["Sample Width (bytes)"]
    sample_width2 = header_info2["Sample Width (bytes)"]

    samples1 = get_samples(raw_data1, sample_width1)
    samples2 = get_samples(raw_data2, sample_width2)

    normalized_values1 = hex_to_normalized_decimal(raw_data1, sample_width1)
    normalized_values2 = hex_to_normalized_decimal(raw_data2, sample_width2)

    differences = calculate_differences(normalized_values1, normalized_values2)

    print(f"First 10 hexadecimal samples for {file1}: {[binascii.hexlify(struct.pack('<' + ('h' if sample_width1 == 2 else 'b'), sample)).decode('utf-8') for sample in samples1][:10]}")
    print(f"First 10 hexadecimal samples for {file2}: {[binascii.hexlify(struct.pack('<' + ('h' if sample_width2 == 2 else 'b'), sample)).decode('utf-8') for sample in samples2][:10]}")

    print(f"First 10 normalized decimal values for {file1}: {normalized_values1[:10]}")
    print(f"First 10 normalized decimal values for {file2}: {normalized_values2[:10]}")

    if output_type == "histogram":
        hist, edges = plot_histogram(differences)

        with open("histogram_data.txt", "w") as f:
            f.write("Difference Interval\tFrequency\n")
            for i in range(len(edges) - 1):
                interval = f"{edges[i]:.2f} to {edges[i+1]:.2f}"
                frequency = hist[i]
                f.write(f"{interval}\t{frequency}\n")

        with open("histogram_data.txt", "r") as f:
            print(f.read())
    elif output_type == "difference":
        plot_noise_wave(differences)

        with open("differences_data.txt", "w") as f:
            f.write("Sample Index\tDifference\n")
            for index, difference in enumerate(differences):
                f.write(f"{index}\t{difference:.6f}\n")

        
        output_wav_file = r"C:\Users\anana\OneDrive\Desktop\compare_wav\differences.wav"
        sample_rate = header_info1["Frame Rate (samples/sec)"]
        sample_width = sample_width1

        print(f"Attempting to write differences to: {output_wav_file}")

        write_difference_wav(differences, output_wav_file, sample_rate, sample_width)
        print(f"Differences written to {output_wav_file}")
        
        if os.path.exists(output_wav_file):
            print(f"File {output_wav_file} successfully created.")
        else:
            print(f"File {output_wav_file} was not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two WAV files and output a histogram or noise wave.")
    parser.add_argument("-audio1", required=True, help="Path to the first audio file")
    parser.add_argument("-audio2", required=True, help="Path to the second audio file")
    parser.add_argument("-output", required=True, choices=["histogram", "difference"], help="Type of output: histogram or difference")
    
    args = parser.parse_args()
    compare_wav_files(args.audio1, args.audio2, args.output)
