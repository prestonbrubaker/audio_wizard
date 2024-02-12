import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from scipy.ndimage import zoom

def linear_spectrogram_to_audio(input_image_path, output_audio_path, sr=44100, n_iter=32, hop_length=512, n_fft=2048):
    print(f"Inverting {input_image_path}...")

    # Load the spectrogram image and convert it
    img = plt.imread(input_image_path)
    if img.ndim == 3:  # Convert RGB to grayscale if necessary
        img = np.mean(img, axis=2)
    img_db = img / np.max(img) * 80 - 80  # Convert pixel values back to dB
    S_db = img_db

    # Convert dB back to amplitude
    S_mag = librosa.db_to_amplitude(S_db)
    
    # Assuming the spectrogram might need to be resized correctly for the inversion
    # Check if resizing is necessary based on your actual use case
    # Calculate the desired shape based on n_fft
    desired_shape = (n_fft // 2 + 1, S_mag.shape[1])
    zoom_factors = (desired_shape[0] / S_mag.shape[0], 1)  # Calculate zoom factors
    S_mag_adjusted = zoom(S_mag, zoom_factors, order=1)  # Apply zoom

    # Invert using Griffin-Lim algorithm with adjusted S_mag
    y_reconstructed = librosa.griffinlim(S_mag_adjusted, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)

    # Save the reconstructed waveform
    wav_output_path = output_audio_path.replace('.mp3', '.wav')
    sf.write(wav_output_path, y_reconstructed, sr)
    print(f"Saved waveform to {wav_output_path}")

    # Optionally convert to MP3, commenting out to retain WAV for quality check
    # You can uncomment if MP3 is needed
    # audio = AudioSegment.from_wav(wav_output_path)
    # audio.export(output_audio_path, format="mp3")
    # print(f"Converted and saved MP3 to {output_audio_path}")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            # Adjusting to save .wav files for quality inspection
            output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')
            linear_spectrogram_to_audio(input_image_path, output_audio_path, sr=44100, n_iter=32, hop_length=512, n_fft=2048)

# Example usage
input_folder = 'greyscales'
output_folder = 'test_gen_mp3s'
process_folder(input_folder, output_folder)
