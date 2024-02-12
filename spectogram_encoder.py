import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def mp3_to_mel_spectrogram(mp3_path, save_path_grayscale, debug_log_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    try:
        y, sr = librosa.load(mp3_path, sr=sr, mono=True)  # Load the audio file
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)  # Compute the mel spectrogram
        S_DB = librosa.power_to_db(S, ref=np.max)  # Convert to decibel scale
        
        # Check if S_DB has variation to avoid division by zero
        if S_DB.max() > S_DB.min():
            S_DB_normalized = 2 * ((S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())) - 1  # Normalize the spectrogram to [-1, 1]
        else:
            # Log debugging information
            with open(debug_log_path, "a") as log_file:
                log_file.write(f"Normalization issue in file: {mp3_path}. S_DB.max() = {S_DB.max()}, S_DB.min() = {S_DB.min()}\n")
            # Handle the no-variation scenario - Consider setting to zeros or another appropriate value
            S_DB_normalized = np.zeros_like(S_DB)

        # Create Grayscale figure
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB_normalized, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='gray')
        plt.axis('off')
        plt.savefig(save_path_grayscale, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        # Log any exception that occurs
        with open(debug_log_path, "a") as log_file:
            log_file.write(f"Error processing file: {mp3_path}. Error: {str(e)}\n")

def process_folder(input_folder, grayscale_folder='copygray', debug_log_folder='debug_logs', file_extension=".mp3"):
    os.makedirs(grayscale_folder, exist_ok=True)
    os.makedirs(debug_log_folder, exist_ok=True)
    debug_log_path = os.path.join(debug_log_folder, "debug_log.txt")
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(file_extension):
            mp3_path = os.path.join(input_folder, filename)
            image_name = filename.replace(file_extension, ".png")
            save_path_grayscale = os.path.join(grayscale_folder, image_name)
            mp3_to_mel_spectrogram(mp3_path, save_path_grayscale, debug_log_path)
            print(f"Processed {filename}: Grayscale spectrograms saved.")

# Example usage
input_folder = "copya"
process_folder(input_folder)
