import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def encode_phase(phase_matrix):
    # Normalize phase to [0, 1] for image representation
    encoded_phase = (phase_matrix + np.pi) / (2 * np.pi)
    return encoded_phase

def create_combined_mel_spectrogram_image(file_path, output_folder, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr)
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # Magnitude and phase
    magnitude, phase = librosa.magphase(D)
    # Mel spectrogram
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_magnitude = np.dot(mel_filter, np.abs(D)**2)
    # Log scale magnitude for better visualization
    log_mel_magnitude = librosa.power_to_db(mel_magnitude, ref=np.max)
    # Encode phase
    encoded_phase = encode_phase(np.angle(D))
    # Combine encoded phase and magnitude in one image (stacked or side-by-side)
    combined_image = np.vstack([log_mel_magnitude, encoded_phase])  # Example: vertical stack

    # Plot
    fig, ax = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)
    img1 = librosa.display.specshow(log_mel_magnitude, y_axis='mel', x_axis='time', ax=ax[0])
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    ax[0].set(title='Mel Spectrogram (Magnitude)')
    img2 = librosa.display.specshow(encoded_phase, cmap='hsv', y_axis='mel', x_axis='time', ax=ax[1])
    fig.colorbar(img2, ax=ax[1])
    ax[1].set(title='Encoded Phase')
    plt.tight_layout()

    # Save as image
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_combined.png'
    plt.savefig(os.path.join(output_folder, output_filename))
    plt.close()

def process_audio_files(folder_path, output_folder):
    for file in os.listdir(folder_path):
        if file.endswith('.mp3'):
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file}")
            create_combined_mel_spectrogram_image(file_path, output_folder)

# Example usage
input_folder = "copy"
output_folder = "copyspecto"
process_audio_files(input_folder, output_folder)
