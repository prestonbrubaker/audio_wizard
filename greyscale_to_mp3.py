import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

def spectrogram_to_audio(input_image_path, output_audio_path, sr=22050, n_iter=32, hop_length=256, db_range=80):
    # Load the spectrogram image
    img = plt.imread(input_image_path)
    if img.ndim == 3:  # Convert RGB to grayscale if necessary
        img = np.mean(img, axis=2)
    # Normalize image values to [0, 1] if not already
    img = img / np.max(img)
    
    # Reverse the dB scale conversion
    img_db = img * db_range - db_range  # Assuming the min dB value was -db_range
    S_mag = librosa.db_to_power(img_db)  # Convert dB back to magnitude

    # Initialize with a random phase
    phase = np.exp(2j * np.pi * np.random.rand(*S_mag.shape))
    S_complex = S_mag * phase

    # Griffin-Lim algorithm
    for _ in range(n_iter):
        audio = librosa.istft(S_complex, hop_length=hop_length)
        _, phase = librosa.magphase(librosa.stft(audio, n_fft=(S_mag.shape[0]*2)-1, hop_length=hop_length))
        S_complex = S_mag * phase

    # Inverse STFT to get time-domain signal
    y = librosa.istft(S_complex, hop_length=hop_length)

    # Save to WAV and then convert to MP3
    wav_path = output_audio_path.rsplit(".", 1)[0] + ".wav"
    sf.write(wav_path, y, sr)
    audio = AudioSegment.from_wav(wav_path)
    audio.export(output_audio_path, format="mp3")
    os.remove(wav_path)

def process_folder(input_folder, output_folder, sr=22050, n_iter=32, hop_length=256, db_range=80):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_image_path = os.path.join(input_folder, filename)
            output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mp3')
            spectrogram_to_audio(input_image_path, output_audio_path, sr, n_iter, hop_length, db_range)

# Example usage
input_folder = 'greyscales'
output_folder = 'test_gen_mp3s'
process_folder(input_folder, output_folder)
