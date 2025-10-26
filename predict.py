import os
import torch
import librosa
from dataset import SoundDataset, AudioCNN, MAX_AUDIO_LEN

# -------------------------
# Settings
# -------------------------
DATA_DIR = "./"  # root folder containing class subfolders
MODEL_PATH = "audio_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load dataset info for labels
# -------------------------
dataset = SoundDataset(DATA_DIR)
label2idx = dataset.label2idx
idx2label = {v: k for k, v in label2idx.items()}
num_classes = len(label2idx)

# -------------------------
# Load model
# -------------------------
model = AudioCNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Audio loading function
# -------------------------
def load_audio_fixed(path, sr=16000):
    try:
        waveform, _ = librosa.load(path, sr=sr, mono=True)
        waveform = torch.tensor(waveform)
        # Pad or truncate to MAX_AUDIO_LEN
        if len(waveform) < MAX_AUDIO_LEN:
            waveform = torch.cat([waveform, torch.zeros(MAX_AUDIO_LEN - len(waveform))])
        else:
            waveform = waveform[:MAX_AUDIO_LEN]
        return waveform.unsqueeze(0).unsqueeze(0).float()  # [1, 1, MAX_AUDIO_LEN]
    except Exception as e:
        print(f"Could not process {path}: {e}")
        return None

# -------------------------
# Prediction
# -------------------------
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        print(f"\nFolder: {folder}")
        for file in os.listdir(folder_path):
            if file.endswith(".wav") or file.endswith(".mp3"):
                file_path = os.path.join(folder_path, file)
                waveform = load_audio_fixed(file_path)
                if waveform is None:
                    continue
                waveform = waveform.to(DEVICE)
                with torch.no_grad():
                    output = model(waveform)
                    pred = torch.argmax(output, dim=1).item()
                    label_name = idx2label[pred]
                    print(f"{file} --> {label_name}")
