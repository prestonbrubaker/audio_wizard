import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

def mp3_to_complex_spectrogram(input_folder, output_folder, sr=44100, n_fft=4096, hop_length=256):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(path).set_channels(1)
            samples = np.array(audio.get_array_of_samples())
            
            # Ensure consistent sampling rate
            y = librosa.resample(np.array(samples, dtype=float), orig_sr=audio.frame_rate, target_sr=sr)
            
            # Ensure fixed duration
            target_length = 10 * sr
            y = librosa.util.fix_length(data=y, size=target_length)
            
            # Generate complex spectrogram
            S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann', center=False)
            
            # Plot the magnitude spectrogram
            S_mag = np.abs(S)
            plt.figure(figsize=(20, 8))  # Larger figure size
            librosa.display.specshow(librosa.amplitude_to_db(S_mag, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            print(f"Processed {filename}: Output Path={output_path}")

# Example usage
input_folder = "mp3_folder"
output_folder = "complex_spectrograms"
mp3_to_complex_spectrogram(input_folder, output_folder)