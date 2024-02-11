import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms

# Define the Autoencoder architecture
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        # The encoder compresses the input into a smaller latent representation.
        # Linear layers are used here, but for more complex data or better performance,
        # convolutional layers could be more appropriate.
        self.encoder = nn.Sequential(
            nn.Linear(128 * 44, 1024),  # Example input size (flattened Mel Spectrogram)
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, 64)  # Latent representation size
        )
        
        # The decoder attempts to reconstruct the input from the latent representation.
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * 44),  # Output size matches input size
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def forward(self, x):
        # Forward pass through encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def preprocess_audio(file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Convert waveform to Mel Spectrogram
    mel_spectrogram_transform = transforms.MelSpectrogram(sample_rate)
    mel_spectrogram = mel_spectrogram_transform(waveform)
    
    # Convert the Mel Spectrogram to decibels for better representation
    db_transform = transforms.AmplitudeToDB()
    mel_spectrogram_db = db_transform(mel_spectrogram)
    
    return mel_spectrogram_db

# Initialize the autoencoder, loss criterion, and optimizer
autoencoder = AudioAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

# Path to the directory containing MP3 files
directory_path = 'path/to/your/directory'

# Iterate over all MP3 files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".mp3"):
        file_path = os.path.join(directory_path, filename)
        
        # Preprocess the file to obtain Mel Spectrogram in dB
        mel_spectrogram_db = preprocess_audio(file_path)
        
        # Assuming the spectrogram shape is [1, 128, 44] after squeezing batch dim
        # Flatten the Mel Spectrogram to fit the autoencoder's input layer
        mel_spectrogram_db_flat = mel_spectrogram_db.view(1, -1)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Pass the spectrogram through the autoencoder
        outputs = autoencoder(mel_spectrogram_db_flat)
        
        # Calculate loss
        loss = criterion(outputs, mel_spectrogram_db_flat)
        
        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
        # Print loss for debugging
        print(f'Loss for {filename}: {loss.item()}')
