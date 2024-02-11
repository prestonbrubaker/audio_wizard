import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_mel_spectrogram(mp3_path, save_path_rgb, save_path_grayscale, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Load the audio file
    y, sr = librosa.load(mp3_path, sr=sr, mono=True)
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to decibel scale for visual representation
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Create RGB figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path_rgb, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Create Grayscale figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path_grayscale, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, rgb_folder='rgb', grayscale_folder='grayscale', file_extension=".mp3"):
    # Ensure output folders exist, create if they don't
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(grayscale_folder, exist_ok=True)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(file_extension):
            mp3_path = os.path.join(input_folder, filename)
            image_name = filename.replace(file_extension, ".png")
            save_path_rgb = os.path.join(rgb_folder, image_name)
            save_path_grayscale = os.path.join(grayscale_folder, image_name)
            mp3_to_mel_spectrogram(mp3_path, save_path_rgb, save_path_grayscale)
            print(f"Processed {filename}: RGB and Grayscale spectrograms saved.")

# Example usage
input_folder = "flat_data"
process_folder(input_folder)
