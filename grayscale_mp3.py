import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from skimage.io import imread

def png_mel_spectrogram_to_audio(image_path, output_audio_path, sr=44100, n_fft=2048, hop_length=512, n_mels=128, n_iter=32):
    # Load the Mel spectrogram image
    S_DB_image = imread(image_path)
    S_DB_normalized = S_DB_image.astype(np.float32) / 255.0  # Convert pixel values to [0, 1]

    # Reverse the normalization to [-1, 1]
    S_DB_normalized = (S_DB_normalized * 2) - 1  # Convert [0, 1] to [-1, 1]

    # Assuming the original min and max dB values are needed but unknown, you might need to estimate or standardize these values.
    # Reverse any other preprocessing (e.g., decibel conversion) based on your original spectrogram settings

    # Convert the Mel spectrogram from decibels back to power
    S_power = librosa.db_to_power(S_DB_normalized)  # Adjust this line based on your actual preprocessing steps

    # Inverse Mel transformation
    S = librosa.feature.inverse.mel_to_stft(S_power, sr=sr, n_fft=n_fft)

    # Use the Griffin-Lim algorithm to estimate phase information
    y = librosa.griffinlim(S, hop_length=hop_length, n_iter=n_iter)

    # Save the reconstructed audio
    librosa.output.write_wav(output_audio_path, y, sr)

# Example usage
image_path = 'path_to_your_mel_spectrogram.png'  # Update this path
output_audio_path = 'reconstructed_audio.wav'  # Update this path
png_mel_spectrogram_to_audio(image_path, output_audio_path)
