import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

def spectrogram_to_audio(input_image_path, output_audio_path, sr=22050, n_iter=32, hop_length=512):
    img = plt.imread(input_image_path)
    if img.ndim == 3:  
        img = np.mean(img, axis=2)
    img = img / np.max(img)

    # Assuming the image represents a log-mel spectrogram, reverse the conversion
    img_db = img * 80 - 80  
    S_mag = librosa.db_to_power(img_db)

    # We skip direct frequency bin adjustment and focus on adjusting dimensions for ISTFT
    n_fft = 2048  # Adjust based on the expected frequency resolution and STFT assumptions

    # Initialize with a random phase for Griffin-Lim
    phase = np.exp(2j * np.pi * np.random.rand(*S_mag.shape))
    S_complex = S_mag * phase

    for _ in range(n_iter):
        audio = librosa.istft(S_complex, hop_length=hop_length, win_length=n_fft)
        _, phase = librosa.magphase(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_complex = S_mag * phase

    y = librosa.istft(S_complex, hop_length=hop_length, win_length=n_fft)

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
