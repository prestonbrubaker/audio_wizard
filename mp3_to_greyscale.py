from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def mp3_to_spectrogram(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(path)
            audio_mono = audio.set_channels(1)  # Convert to mono
            samples = np.array(audio_mono.get_array_of_samples())
            sample_rate = audio.frame_rate

            # Ensure fixed duration for consistent spectrogram length
            standard_duration = 10  # seconds
            target_samples = standard_duration * sample_rate
            if len(samples) > target_samples:
                samples = samples[:target_samples]
            elif len(samples) < target_samples:
                samples = np.pad(samples, (0, target_samples - len(samples)), 'constant')

            # Convert samples to librosa format
            y = librosa.util.buf_to_float(samples, dtype=np.float32)
            
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

            # Print audio file information to console
            duration_seconds = len(samples) / sample_rate
            print(f"Processed {filename}: Duration={duration_seconds:.2f}s, Sample Rate={sample_rate}Hz, Channels=1, Output Path={output_path}")

# Example usage
input_folder = 'flat_data'
output_folder = 'greyscales'
mp3_to_spectrogram(input_folder, output_folder)
