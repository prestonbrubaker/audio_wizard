import os
import numpy as np
import librosa
import imageio.v2 as imageio
from scipy.io.wavfile import write

def grayscale_image_to_audio(image_path, output_audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    # Load the grayscale spectrogram image
    img = imageio.imread(image_path)
    img_normalized = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    # Map pixel values back to decibels
    # Assuming the dynamic range is -80 dB to 0 dB
    S_DB = img_normalized * 80.0 - 80.0  # Inverse operation of librosa.power_to_db

    # Convert decibels back to power
    S_power = librosa.db_to_power(S_DB)

    # Invert the mel spectrogram back to audio
    y = librosa.feature.inverse.mel_to_audio(S_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Save the inverted audio to a WAV file
    wav_output_path = output_audio_path.replace('.mp3', '.wav')
    write(wav_output_path, sr, (y * 32767).astype(np.int16))  # Scale float to int16 for WAV format

    # Convert WAV to MP3
    os.system(f'ffmpeg -i "{wav_output_path}" -codec:a libmp3lame -qscale:a 2 "{output_audio_path}"')

    # Optionally, remove the WAV file after conversion to MP3
    os.remove(wav_output_path)

def process_folder(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            mp3_output_path = os.path.join(output_folder, filename.replace('.png', '.mp3'))
            grayscale_image_to_audio(image_path, mp3_output_path)
            print(f"Converted {filename} to audio.")

# Example usage
input_folder = "grayscale"
output_folder = "gray_mp3"
process_folder(input_folder, output_folder)
