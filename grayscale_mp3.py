import os
import numpy as np
import librosa
import imageio.v2 as imageio
from scipy.io.wavfile import write

def image_to_audio(image_path, save_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, dynamic_range=80):
    """
    Convert a grayscale spectrogram image back into an audio file.

    Args:
    - image_path: Path to the input grayscale spectrogram image.
    - save_path: Path to save the output audio file.
    - sr, n_fft, hop_length, n_mels: Audio and spectrogram parameters.
    - dynamic_range: The dynamic range of the spectrogram in decibels.
    """
    # Load the grayscale image
    img = imageio.imread(image_path)

    # Normalize the image data to [0, 1] range
    img_normalized = img.astype(np.float32) / 255.0

    # Map normalized image data to decibel scale
    S_DB = img_normalized * dynamic_range - dynamic_range  # Mapping [0, 1] -> [-80 dB, 0 dB]

    # Convert decibel scale to power
    S_power = librosa.db_to_power(S_DB)

    # Invert the mel spectrogram to audio
    # Note: librosa's mel_to_audio might not need n_mels explicitly, it's determined from S_power shape
    y = librosa.feature.inverse.mel_to_audio(S_power, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

    # Save the audio to a WAV file first (Librosa works with floats, but scipy.io.wavfile expects int16 format)
    wav_path = save_path.replace('.mp3', '.wav')
    write(wav_path, sr, (y * 32767).astype(np.int16))

    # Convert WAV to MP3 using ffmpeg
    os.system(f'ffmpeg -i "{wav_path}" -codec:a libmp3lame -qscale:a 2 "{save_path}"')

    # Optionally, remove the WAV file after conversion
    os.remove(wav_path)

def process_folder(input_folder, output_folder):
    """
    Convert all grayscale spectrogram images in a folder back into MP3 audio files.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            mp3_name = filename.replace(".png", ".mp3")
            save_path = os.path.join(output_folder, mp3_name)
            image_to_audio(image_path, save_path)
            print(f"Converted {filename} back to audio and saved as {mp3_name}.")

# Example usage
input_folder = "grayscale"
output_folder = "grayscale_mp3"
process_folder(input_folder, output_folder)
