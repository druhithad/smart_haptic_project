import serial
import time
import torch
import torch.nn as nn
from dataset import AudioCNN, MAX_AUDIO_LEN
import json

# -------------------------
# Settings
# -------------------------
SERIAL_PORT = 'COM6'        # ESP32 port
BAUD_RATE = 115200
MODEL_PATH = 'audio_cnn.pth'
LABEL_MAP_PATH = 'label_map.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRIGGER_THRESHOLD = 400     # mic threshold for detecting sound

# -------------------------
# Load labels
# -------------------------
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
idx2label = {v: k for k, v in label_map.items()}
num_classes = len(idx2label)

# -------------------------
# Load model
# -------------------------
model = AudioCNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------
# Vibration map (pulses per sound)
# -------------------------
VIBRATION_MAP = {
    'animal sounds': 1,
    'car sounds': 2,
    'disturbance sounds': 1,
    'horn sounds': 2,
    'human sounds': 1,
    'random sounds': 1,
    'siren sounds': 3,
    'unknown sounds': 1
}

# -------------------------
# Connect to ESP32
# -------------------------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("Listening to ESP32... Press Ctrl+C to stop")

# -------------------------
# Prediction function
# -------------------------
def predict_sound(value):
    """
    Convert a single mic value to a tensor for model prediction.
    Pads/truncates to MAX_AUDIO_LEN.
    """
    waveform = torch.zeros(MAX_AUDIO_LEN)
    waveform[0] = value  # single value at start, rest zeros
    waveform = waveform.unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # [1,1,MAX_AUDIO_LEN]

    with torch.no_grad():
        output = model(waveform)
        pred_idx = torch.argmax(output, dim=1).item()
        label_name = idx2label.get(pred_idx, 'unknown')
    return label_name

# -------------------------
# Main loop
# -------------------------
try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            try:
                mic_val = int(line)
            except:
                continue

            # Trigger detection
            if mic_val > TRIGGER_THRESHOLD:
                detected_sound = predict_sound(mic_val)
                pulses = VIBRATION_MAP.get(detected_sound, 1)
                print(f"[DETECTED] Sound: {detected_sound} | Vibrating: {pulses} pulse(s)", flush=True)

except KeyboardInterrupt:
    print("\nStopped by user")
    ser.close()
