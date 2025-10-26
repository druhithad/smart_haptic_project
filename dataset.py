import os
import torch
from torch.utils.data import Dataset
import librosa

MAX_AUDIO_LEN = 16000 * 5  # 5 seconds at 16kHz

def load_audio(path, sr=16000):
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    # Pad or truncate
    if len(waveform) < MAX_AUDIO_LEN:
        waveform = torch.cat([torch.tensor(waveform), torch.zeros(MAX_AUDIO_LEN - len(waveform))])
    else:
        waveform = torch.tensor(waveform[:MAX_AUDIO_LEN])
    return waveform

class SoundDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = {}
        self.label2idx = {}
        current_label = 0

        # Walk through folders
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                self.label2idx[folder] = current_label
                for file in os.listdir(folder_path):
                    if file.endswith(".wav") or file.endswith(".mp3"):
                        self.data.append((os.path.join(folder_path, file), current_label))
                current_label += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        waveform = load_audio(path)
        return waveform.unsqueeze(0).float(), label  # add channel dimension

# Simple CNN model
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 9, stride=1, padding=4)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, 9, stride=1, padding=4)
        self.fc1 = nn.Linear(32 * (MAX_AUDIO_LEN // 16), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
