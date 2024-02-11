import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

def spectrogram_to_audio(input_image_path, output_audio_path, sr=22050, n_iter=32, hop_length=256):
    # Load the spectrogram image
    img = plt.imread(input_image_path)
    if img.ndim == 3:  # Convert RGB to grayscale if necessary
        img = np.mean(img, axis=2)
    img = img / np.max(img)  # Normalize image values to [0, 1]
    
    # Convert image pixel values to power spectrogram (assuming linear scale for simplicity)
    img_db = img * 80 - 80  # Assuming the min dB value was -80 dB
    S_mag = librosa.db_to_power(img_db)  # Convert dB back to power

    # Generate a random phase array with the same shape as S_mag
    phase = np.exp(2j * np.pi * np.random.rand(*S_mag.shape))

    # Initialize an empty complex spectrogram with the correct shape
    S_complex = np.zeros(S_mag.shape, dtype=complex)  # Updated to use 'complex'

    # Griffin-Lim algorithm to estimate phase
    for i in range(n_iter):
        # Combine magnitude and phase to get the complex spectrogram
        S_complex = S_mag * np.exp(1j * np.angle(phase))
        # Inverse STFT
        y = librosa.istft(S_complex, hop_length=hop_length)
        # Recompute the STFT to get a new phase estimate
        _, phase = librosa.magphase(librosa.stft(y, n_fft=(S_mag.shape[0]-1)*2, hop_length=hop_length))

    # Reconstruct audio from the magnitude and estimated phase
    y_reconstructed = librosa.istft(S_complex, hop_length=hop_length)

    # Save the reconstructed signal to a WAV file first
    wav_path = output_audio_path.rsplit(".", 1)[0] + ".wav"  # Change the extension to .wav
    sf.write(wav_path, y_reconstructed, sr)

    # Convert the WAV file to MP3 using pydub
    audio = AudioSegment.from_wav(wav_path)
    audio.export(output_audio_path, format="mp3")

    # Optionally, remove the temporary WAV file to save space
    os.remove(wav_path)
    
# Example usage
input_folder = 'greyscales'
output_folder = 'reconstructed_mp3s'
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_image_path = os.path.join(input_folder, filename)
        output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mp3')
        spectrogram_to_audio(input_image_path, output_audio_path, sr=22050, n_iter=32, hop_length=256)

