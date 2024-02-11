import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
from pydub import AudioSegment

def spectrogram_to_audio(input_image_path, output_audio_path, sr=22050, n_iter=32, hop_length=256):
    print(f"Processing {input_image_path}...")

    # Load the spectrogram image
    img = plt.imread(input_image_path)
    print(f"Image shape (HxW): {img.shape}")
    if img.ndim == 3:  # Convert RGB to grayscale if necessary
        img = np.mean(img, axis=2)
    img = img / np.max(img)  # Normalize image values to [0, 1]
    
    # Assuming the conversion scale might need adjustment; 
    # this is a placeholder for actual values based on how the spectrogram was generated
    img_db = img * 80 - 80
    S_mag = librosa.db_to_power(img_db)

    # Generate a random phase array with the same shape as S_mag
    phase = np.exp(2j * np.pi * np.random.rand(*S_mag.shape))

    # Initialize an empty complex spectrogram
    S_complex = np.zeros(S_mag.shape, dtype=complex)

    # Griffin-Lim algorithm to estimate phase
    for i in range(n_iter):
        S_complex = S_mag * np.exp(1j * np.angle(phase))
        y = librosa.istft(S_complex, hop_length=hop_length)
        _, phase = librosa.magphase(librosa.stft(y, n_fft=(S_mag.shape[0]-1)*2, hop_length=hop_length))

    # Reconstruct audio from the magnitude and estimated phase
    y_reconstructed = librosa.istft(S_complex, hop_length=hop_length)
    # Normalize the amplitude of the reconstructed audio
    y_reconstructed = y_reconstructed / np.max(np.abs(y_reconstructed))

    # Before saving the WAV file, check the max amplitude
    max_amp = np.max(np.abs(y_reconstructed))
    print(f"Max amplitude of reconstructed audio: {max_amp}")
    if max_amp < 1e-3:  # This threshold is arbitrary; adjust based on your observations
        print("Warning: Reconstructed audio might be too quiet.")
    # Save the reconstructed signal to a WAV file first
    wav_path = output_audio_path.rsplit(".", 1)[0] + ".wav"
    sf.write(wav_path, y_reconstructed, sr)
    print(f"Saved WAV to {wav_path}")

    # Convert the WAV file to MP3 using pydub
    audio = AudioSegment.from_wav(wav_path)
    audio.export(output_audio_path, format="mp3")
    print(f"Converted and saved MP3 to {output_audio_path}")

    # Optionally, remove the temporary WAV file to save space
    os.remove(wav_path)
    print("Temporary WAV file removed.")
    
# Example usage
input_folder = 'greyscales'
output_folder = 'test_gen_mp3s'
# Process each spectrogram in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_image_path = os.path.join(input_folder, filename)
        output_audio_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.mp3')
        spectrogram_to_audio(input_image_path, output_audio_path, sr=22050, n_iter=32, hop_length=256)
