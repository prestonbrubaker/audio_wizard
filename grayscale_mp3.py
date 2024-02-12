import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from scipy.ndimage import zoom

def adjust_spectrogram_volume(S_dB, target_dB=-5):
    """
    Adjusts the spectrogram volume by a target dB level.
    """
    return S_dB + target_dB  # Only adjust by the target dB level

def linear_spectrogram_to_audio(input_image_path, output_audio_path, sr=44100, n_iter=64, hop_length=512, n_fft=2048):
    print(f"Inverting {input_image_path}...")

    img = plt.imread(input_image_path)
    if img.ndim == 3:
        img = np.mean(img, axis=2)
    img_db = img * 80 - 80  # Remove the normalization step

    # Adjust the volume of the spectrogram
    img_db_adjusted = adjust_spectrogram_volume(img_db, target_dB=-5)
    S_mag = librosa.db_to_amplitude(img_db_adjusted)

    desired_shape = (n_fft // 2 + 1, int(sr * 10 / hop_length))  # Calculate desired shape for 10-second audio
    S_mag_adjusted = zoom(S_mag, (desired_shape[0] / S_mag.shape[0], desired_shape[1] / S_mag.shape[1]), order=1)

    y_reconstructed = librosa.griffinlim(S_mag_adjusted, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)

    # Adjust the length of the signal to achieve a duration of 10 seconds
    y_reconstructed = y_reconstructed[:sr * 10]

    sf.write(output_audio_path, y_reconstructed, sr)  # Write the signal to the output file
    print(f"Saved audio to {output_audio_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')
            linear_spectrogram_to_audio(input_image_path, output_audio_path)

input_folder = 'grayscale'
output_folder = 'test1'
process_folder(input_folder, output_folder)
