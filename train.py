import torch
from torch.utils.data import DataLoader, random_split
from dataset import SoundDataset, AudioCNN
DATA_DIR = "./"  # root folder containing all class folders
  # or the folder containing subfolders
BATCH_SIZE = 2
EPOCHS = 5
LR = 0.001

# Load dataset
dataset = SoundDataset(DATA_DIR)
num_classes = len(dataset.label2idx)

# Train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN(num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "audio_cnn.pth")
print("Training finished. Model saved as audio_cnn.pth")
