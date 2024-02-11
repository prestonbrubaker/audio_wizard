import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn









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








def load_mp3(filename):
    waveform, sample_rate = torchaudio.load(filename)
    return waveform, sample_rate
from torchaudio.transforms import MelSpectrogram

def to_mel_spectrogram(waveform, sample_rate, n_mels=128):
    mel_spectrogram = MelSpectrogram(sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram
def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor.abs().max()


class AudioDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = load_mp3(self.file_paths[idx])
        # Choose preprocessing method: to_spectrogram or to_mel_spectrogram
        processed_data = to_mel_spectrogram(waveform, sample_rate)
        normalized_data = normalize(processed_data)
        return normalized_data

dataset = AudioDataset(['path/to/your/mp3file1.mp3', 'path/to/your/mp3file2.mp3'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



