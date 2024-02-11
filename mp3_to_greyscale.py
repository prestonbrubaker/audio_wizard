import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

def mp3_to_spectrogram(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            # Load the MP3 file
            path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(path)
            audio = audio.set_channels(1)  # Convert to mono
            samples = np.array(audio.get_array_of_samples())
            
            # Convert samples to librosa format
            y = librosa.util.buf_to_float(samples, dtype=np.float32)
            sample_rate = audio.frame_rate
            
            # Generate spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Create a figure without any axes
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sample_rate, cmap='gray_r', fmax=8000)
            
            # Remove axes and legends
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Save the spectrogram as a grayscale PNG
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

# Example usage
input_folder = 'flat_data'
output_folder = 'greyscales'
mp3_to_spectrogram(input_folder, output_folder)
