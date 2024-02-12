import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from skimage.io import imread

def png_mel_spectrogram_to_audio(image_path, output_audio_path, sr=44100, n_fft=2048, hop_length=512, n_iter=32):
    # Load the Mel spectrogram image
    S_DB_image = imread(image_path)
    S_DB_normalized = S_DB_image.astype(np.float32)  # No normalization needed

    # Convert the Mel spectrogram from decibels back to power
    S_power = librosa.db_to_power(S_DB_normalized)

    # Inverse Mel transformation
    S = librosa.feature.inverse.mel_to_stft(S_power, sr=sr, n_fft=n_fft)

    # Use the Griffin-Lim algorithm to estimate phase information
    y = librosa.griffinlim(S, hop_length=hop_length, n_iter=n_iter)

    # Save the reconstructed audio
    librosa.output.write_wav(output_audio_path, y, sr)

def process_folder(input_folder, output_folder, sr=44100, n_fft=2048, hop_length=512, n_iter=32):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')
            
            print(f"Processing {filename}...")
            png_mel_spectrogram_to_audio(image_path, output_audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_iter=n_iter)
            print(f"Finished processing {filename}. Audio saved to {output_audio_path}")

# Example usage
input_folder = 'grayscale'  # Update this path
output_folder = 'test1will'  # Update this path
process_folder(input_folder, output_folder)
