import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_mel_spectrogram(mp3_path, save_path_grayscale, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    y, sr = librosa.load(mp3_path, sr=sr, mono=True)    # Load the audio file
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)    # Compute the mel spectrogram
    S_DB = librosa.power_to_db(S, ref=np.max)    # Convert to decibel scale for visual representation
    S_DB_normalized = 2 * ((S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())) - 1    # Normalize the spectrogram to [-1, 1]

    # Create Grayscale figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB_normalized, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path_grayscale, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, grayscale_folder='grayscale', file_extension=".mp3"):
    os.makedirs(grayscale_folder, exist_ok=True)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(file_extension):
            mp3_path = os.path.join(input_folder, filename)
            image_name = filename.replace(file_extension, ".png")
            save_path_grayscale = os.path.join(grayscale_folder, image_name)
            mp3_to_mel_spectrogram(mp3_path, save_path_grayscale)
            print(f"Processed {filename}: Grayscale spectrograms saved.")

# Example usage
input_folder = "copya"
process_folder(input_folder)
