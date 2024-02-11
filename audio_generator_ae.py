import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
import os
import glob

# Assuming the AudioAutoencoder class definition is the same as before, ensure it's included here or imported if separated into modules

class AudioProcessor:
    def __init__(self, model_path, input_shape, n_mels=128):
        self.model = AudioAutoencoder(input_shape)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set the model to evaluation mode
        self.n_mels = n_mels

    def process_audio(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        mel_spec_processor = MelSpectrogram(sample_rate, n_mels=self.n_mels)
        mel_spectrogram = mel_spec_processor(waveform)
        normalized_mel = normalize(mel_spectrogram)
        return normalized_mel.view(-1)  # Flatten the Mel spectrogram for model input

    def generate_audio(self, input_tensor):
        with torch.no_grad():
            generated_audio = self.model(input_tensor)
        return generated_audio

# Modify these paths and parameters as necessary
model_path = 'autoencoder.pth'
raw_data_folder = 'raw_data'
generated_audio_folder = 'generated_audio'
input_shape = 128*44  # This needs to be the same as used during training

# Create the directory for generated audio if it doesn't exist
os.makedirs(generated_audio_folder, exist_ok=True)

audio_processor = AudioProcessor(model_path, input_shape)

# Process and generate audio for files in raw_data
for file_path in glob.glob(os.path.join(raw_data_folder, '*.mp3'))[:10]:  # Limit to 10 files
    normalized_mel = audio_processor.process_audio(file_path)
    # Assuming the model expects a batch dimension, and single sample needs to be unsqueezed
    generated_audio = audio_processor.generate_audio(normalized_mel.unsqueeze(0))
    # Save the generated audio - you might need additional processing depending on your model's output
    # For example, if your model outputs a Mel spectrogram, you would need an inverse Mel transformation here
    output_path = os.path.join(generated_audio_folder, os.path.basename(file_path))
    # Placeholder: Actual saving method depends on the generated data's format
    # torchaudio.save(output_path, generated_audio, sample_rate)

print("Processing and generation complete.")
