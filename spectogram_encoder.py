import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_mel_spectrogram(mp3_path, save_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Convert an MP3 file to a mel spectrogram and save it as a PNG image using the viridis colormap.

    Args:
    - mp3_path: Path to the input MP3 file.
    - save_path: Path to save the output PNG image.
    - sr: Sampling rate for audio processing.
    - n_fft: FFT window size.
    - hop_length: Number of samples between successive frames.
    - n_mels: Number of Mel bands to generate.
    """
    # Load the audio file
    y, sr = librosa.load(mp3_path, sr=sr, mono=True)
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Convert to decibel scale for visual representation
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Plotting the spectrogram using the viridis colormap
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    plt.close()

def process_folder(input_folder, output_folder, file_extension=".mp3"):
    """
    Process all MP3 files in a folder, converting them to mel spectrograms and saving as images.

    Args:
    - input_folder: Folder containing MP3 files.
    - output_folder: Folder to save spectrogram images.
    - file_extension: Extension of files to process.
    """
    # Ensure output folder exists, create if it doesn't
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(file_extension):
            mp3_path = os.path.join(input_folder, filename)
            image_name = filename.replace(file_extension, ".png")
            save_path = os.path.join(output_folder, image_name)
            mp3_to_mel_spectrogram(mp3_path, save_path)
            print(f"Processed {filename} to {save_path}")

# Example usage
input_folder = "flat_data"
output_folder = "../spectrograms"
process_folder(input_folder, output_folder)
