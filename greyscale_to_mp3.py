import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

def linear_spectrogram_to_audio(input_image_path, output_audio_path, sr=44100, n_iter=32, hop_length=512, n_fft=2048):
    print(f"Inverting {input_image_path}...")

    # Load the spectrogram image
    img = plt.imread(input_image_path)
    if img.ndim == 3:
        img = np.mean(img, axis=2)  # Convert to grayscale
    img_db = img / np.max(img) * 80 - 80  # Assuming max dB range in image is 80 dB
    S_mag = librosa.db_to_amplitude(img_db)
    
    # Invert using Griffin-Lim algorithm
    y_reconstructed = librosa.griffinlim(S_mag, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)

    # Normalize the amplitude
    y_reconstructed = y_reconstructed / np.max(np.abs(y_reconstructed))
    
    sf.write(output_audio_path, y_reconstructed, sr)
    print(f"Saved audio to {output_audio_path}")

# Example usage
input_folder = 'greyscales'
output_folder = 'test_gen_mp3s'
linear_spectrogram_to_audio(os.path.join(input_folder, 'example.png'), os.path.join(output_folder, 'example.mp3'))
