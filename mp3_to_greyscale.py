from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def mp3_to_linear_spectrogram(input_folder, output_folder, sr=44100, n_fft=2048, hop_length=512):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(path).set_channels(1)
            samples = np.array(audio.get_array_of_samples())
            
            # Convert samples to librosa format and ensure consistent sampling rate
            y = librosa.resample(np.array(samples, dtype=float), audio.frame_rate, sr)
            
            # Trim or pad the audio samples to ensure a fixed duration (e.g., 10 seconds)
            target_length = 10 * sr  # 10 seconds at sr
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, max(0, target_length - len(y))), "constant")
            
            # Generate a linear-frequency spectrogram
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            S_dB = librosa.amplitude_to_db(S, ref=np.max)
            
            # Create a figure without any axes for the spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            print(f"Processed {filename}: Output Path={output_path}")

# Example usage
input_folder = 'data_freqmatch'
output_folder = 'linear_spectrograms'
mp3_to_linear_spectrogram(input_folder, output_folder)
