import torch
from torch import nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
import os
import glob

class AudioAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(AudioAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)  # Latent representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_shape),
            nn.Tanh()  # Using Tanh to ensure the output range matches the normalized input range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def normalize(tensor):
    # Scale the tensor to have values between -1 and 1
    return (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 2 - 1

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
        # Use .reshape() instead of .view() for compatibility
        return normalized_mel.reshape(-1)

    def generate_audio(self, input_tensor):
        with torch.no_grad():
            generated_audio = self.model(input_tensor)
        return generated_audio

# Modify these paths and parameters as necessary
model_path = 'autoencoder.pth'
raw_data_folder = 'raw_data'
generated_audio_folder = 'generated_audio'

# Create the directory for generated audio if it doesn't exist
os.makedirs(generated_audio_folder, exist_ok=True)

# Corrected: Use actual_input_shape instead of input_shape
# Assuming the Mel spectrogram size is (n_mels, time_steps)
n_mels = 128
time_steps = 427
actual_input_shape = n_mels * time_steps  # Adjust according to your Mel spectrogram size
audio_processor = AudioProcessor(model_path, actual_input_shape)

# Process and generate audio for files in raw_data
for file_path in glob.glob(os.path.join(raw_data_folder, '*.mp3'))[:10]:  # Limit to 10 files
    normalized_mel = audio_processor.process_audio(file_path)
    print("Shape of normalized_mel:", normalized_mel.shape)  # Add this line
    # Assuming the model expects a batch dimension, and single sample needs to be unsqueezed
    generated_audio = audio_processor.generate_audio(normalized_mel.unsqueeze(0))
    print("Shape before passing to the decoder:", generated_audio.shape)  # Add this line
    # Placeholder for saving generated audio
    output_path = os.path.join(generated_audio_folder, os.path.basename(file_path))
    # Implement actual saving method depending on the generated data's format
    # Example placeholder: torchaudio.save(output_path, generated_audio.squeeze(0), sample_rate)

print("Processing and generation complete.")
print("Processing and generation complete.")
