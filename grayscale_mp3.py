import os
import numpy as np
import librosa
import imageio
from scipy.io.wavfile import write

def image_to_audio(image_path, save_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Convert a grayscale spectrogram image back into an audio file (WAV format).
    
    Args:
    - image_path: Path to the input grayscale spectrogram image.
    - save_path: Path to save the output audio file (WAV format).
    - sr: Sampling rate for audio processing.
    - n_fft: FFT window size.
    - hop_length: Number of samples between successive frames.
    - n_mels: Number of Mel bands.
    """
    # Load the grayscale image
    img = imageio.imread(image_path)
    
    # Normalize the image data to [0, 1] range
    img_normalized = img / 255.0
    
    # Convert normalized image data to decibel scale
    S_DB = img_normalized * -80.0 + 80.0  # Assuming -80 dB as the minimum value
    
    # Convert decibel scale to power
    S_power = librosa.db_to_power(S_DB)
    
    # Inverse mel spectrogram to audio
    y = librosa.feature.inverse.mel_to_audio(S_power, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Save the audio to a WAV file
    wav_path = save_path.replace('.mp3', '.wav')  # Temporary WAV file for intermediate saving
    write(wav_path, sr, y.astype(np.float32))
    
    # Convert WAV to MP3 using ffmpeg
    os.system(f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 {save_path}")
    
    # Optionally, remove the WAV file after conversion
    os.remove(wav_path)

def process_folder(input_folder, output_folder):
    """
    Process all grayscale spectrogram images in a folder, converting them back into MP3 audio files.
    
    Args:
    - input_folder: Folder containing grayscale spectrogram images.
    - output_folder: Folder to save MP3 audio files.
    """
    # Ensure output folder exists, create if it doesn't
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file in the input folder
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
