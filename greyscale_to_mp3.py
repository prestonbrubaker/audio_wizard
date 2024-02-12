import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from scipy.ndimage import zoom

def adjust_spectrogram_volume(S_dB, target_dB=-20):
    """
    Adjusts the spectrogram volume by a target dB level.
    """
    return S_dB - np.max(S_dB) + target_dB

def linear_spectrogram_to_audio(input_image_path, output_audio_path, sr=44100, n_iter=10, hop_length=512, n_fft=2048):
    print(f"Inverting {input_image_path}...")

    img = plt.imread(input_image_path)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    img_db = img / np.max(img) * 80 - 80

    # Adjust the volume of the spectrogram
    img_db_adjusted = adjust_spectrogram_volume(img_db, target_dB=-20)
    S_mag = librosa.db_to_amplitude(img_db_adjusted)

    desired_shape = (n_fft // 2 + 1, S_mag.shape[1])
    zoom_factors = (desired_shape[0] / S_mag.shape[0], 1)
    S_mag_adjusted = zoom(S_mag, zoom_factors, order=1)

    y_reconstructed = librosa.griffinlim(S_mag_adjusted, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)
    sf.write(output_audio_path, y_reconstructed, sr)
    print(f"Saved audio to {output_audio_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')  # Keeping as WAV for quality
            linear_spectrogram_to_audio(input_image_path, output_audio_path)

input_folder = 'greyscales'
output_folder = 'test_gen_mp3s'
process_folder(input_folder, output_folder)
