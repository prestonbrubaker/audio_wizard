import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

def mp3_to_spectrogram(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            # Load the MP3 file
            path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(path)
            audio = audio.set_channels(1)  # Convert to mono
            samples = audio.get_array_of_samples()
            sample_rate = audio.frame_rate
            
            # Convert samples to librosa format
            y = librosa.util.buf_to_float(samples, n_bytes=2, dtype=np.float32)
            
            # Generate spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Plot and save spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', fmax=8000)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-frequency spectrogram')
            plt.tight_layout()
            plt.axis('off')  # Removes axes for a cleaner look
            
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# Example usage
input_folder = 'flat_data'
output_folder = 'greyscales'
mp3_to_spectrogram(input_folder, output_folder)
