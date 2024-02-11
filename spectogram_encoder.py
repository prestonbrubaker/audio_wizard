import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_mel_spectrogram(mp3_path, save_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Load the audio file
    y, sr = librosa.load(mp3_path, sr=sr, mono=True)
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to decibel scale for visual representation
    S_DB = librosa.power_to_db(S, ref=np.max)
    # Create a figure without axes
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='viridis')
    ax.axis('off')  # No axes for a cleaner look
    # Save the figure without borders or axes and close the plot to free memory
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, output_folder, file_extension=".mp3"):
    # Ensure output folder exists, create if it doesn't
    os.makedirs(output_folder, exist_ok=True)
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(file_extension):
            mp3_path = os.path.join(input_folder, filename)
            image_name = filename.replace(file_extension, ".png")
            save_path = os.path.join(output_folder, image_name)
            mp3_to_mel_spectrogram(mp3_path, save_path)
            os.remove(mp3_path)  # Delete the MP3 file after its spectrogram has been saved
            print(f"Transformed {filename} to spectrogram and removed original MP3.")

# Example usage
input_folder = "flat_data"
output_folder = "spectrograms"
process_folder(input_folder, output_folder)