import os
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as transforms
import numpy as np

def create_mel_spectrogram(file_path, sample_rate):
    """
    Converts an audio file to a Mel Spectrogram on a decibel scale,
    averaging channels if the audio is stereo.
    """
    waveform, _ = torchaudio.load(file_path)
    # Check if the audio is stereo (2 channels) and average to mono if necessary
    if waveform.size(0) > 1:  # More than one channel
        waveform = waveform.mean(dim=0, keepdim=True)  # Average channels
    mel_spectrogram_transform = transforms.MelSpectrogram(sample_rate, n_fft=2048, hop_length=512, n_mels=128)
    mel_spectrogram = mel_spectrogram_transform(waveform)
    db_transform = transforms.AmplitudeToDB()
    mel_spectrogram_db = db_transform(mel_spectrogram)
    return mel_spectrogram_db

def save_spectrogram_to_png(spectrogram, output_file_path):
    """
    Saves the Mel Spectrogram to a PNG file with a visually interesting colormap.
    """
    spectrogram_np = spectrogram.squeeze().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_np, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')  # Remove axes for a cleaner look
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_folder(input_folder, output_folder):
    """
    Processes all MP3 files in the input folder, converting them to Mel Spectrograms and saving as PNG images.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            file_path = os.path.join(input_folder, filename)
            mel_spectrogram_db = create_mel_spectrogram(file_path, sample_rate=22050)  # Using a common sample rate for simplicity
            
            # Create the output filename and save the spectrogram as an image
            base_filename = os.path.splitext(filename)[0]
            output_file_path = os.path.join(output_folder, f"{base_filename}.png")
            save_spectrogram_to_png(mel_spectrogram_db, output_file_path)
            print(f"Processed and saved: {output_file_path}")

# Example usage
input_folder = 'raw_data'  # Update this path
output_folder = 'mel_spectograms'  # Update this path
process_folder(input_folder, output_folder)
