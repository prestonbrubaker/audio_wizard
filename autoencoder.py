import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram
import glob
import os




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



def load_mp3(filename, segment_length=10):
    waveform, sample_rate = torchaudio.load(filename)
    num_frames = segment_length * sample_rate  # 10 seconds * sample_rate frames/second
    if waveform.size(1) > num_frames:
        waveform = waveform[:, :num_frames]  # Take the first 'num_frames' frames
    return waveform, sample_rate





def to_mel_spectrogram(waveform, sample_rate, n_mels=128):
    mel_spectrogram = MelSpectrogram(sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram
def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor.abs().max()


class AudioDataset(Dataset):
    def __init__(self, root_dir, segment_length=10):
        self.file_paths = glob.glob(os.path.join(root_dir, '*.mp3'))
        self.segment_length = segment_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = load_mp3(self.file_paths[idx], self.segment_length)
        processed_data = to_mel_spectrogram(waveform, sample_rate)
        normalized_data = normalize(processed_data)
        # Flatten the Mel spectrogram for the autoencoder
        return normalized_data.view(-1)



# Assuming 'raw_data' folder is in the current directory
dataset = AudioDataset('raw_data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Optimizer, and Criterion initialization (example)
actual_input_shape = 128*44  # This needs to be adjusted based on your Mel spectrogram size
model = AudioAutoencoder(input_shape=actual_input_shape)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch.float())  # Ensure the batch is in the correct dtype
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), "autoencoder.pth")
print("Model Saved")


