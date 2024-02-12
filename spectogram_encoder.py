import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def encode_phase(phase_matrix):
    # Normalize phase to [0, 1] for image representation
    return (phase_matrix + np.pi) / (2 * np.pi)

def create_combined_mel_spectrogram_image(file_path, output_folder, sr=44100, n_fft=4096, hop_length=512, n_mels=128):
    y, sr = librosa.load(file_path, sr=sr)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_magnitude = np.dot(mel_filter, np.abs(D)**2)
    log_mel_magnitude = librosa.power_to_db(mel_magnitude, ref=np.max)
    encoded_phase = encode_phase(np.angle(D))
    
    # Prepare plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Mel spectrogram (magnitude) in grayscale
    librosa.display.specshow(log_mel_magnitude, sr=sr, hop_length=hop_length, cmap='gray', ax=ax[0])
    ax[0].axis('off')  # Remove axes for clean image
    
    # Plot encoded phase
    librosa.display.specshow(encoded_phase, cmap='hsv', sr=sr, hop_length=hop_length, ax=ax[1])
    ax[1].axis('off')  # Remove axes for clean image
    
    # Adjust layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Save combined image
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_combined.png'
    plt.savefig(os.path.join(output_folder, output_filename), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def process_audio_files(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in os.listdir(folder_path):
        if file.endswith('.mp3'):
            print(f"Processing file: {file}")
            create_combined_mel_spectrogram_image(os.path.join(folder_path, file), output_folder)


# Example usage
input_folder = "copy"
output_folder = "copyspecto"
process_audio_files(input_folder, output_folder)
