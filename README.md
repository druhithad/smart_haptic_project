# smart_haptic_project
Smart Haptic Project:[for Hearing Impaired] A real-time sound recognition system using ESP32 and Python that triggers haptic feedback based on detected sounds.

# Smart Haptic Project

This project implements a **real-time sound recognition system** using ESP32 and Python. It detects various sound types and provides haptic feedback (vibration) based on the recognized sound.

## Features
- Real-time sound recognition from microphone input
- Supports multiple sound categories (animal, car, horn, siren, human, etc.)
- Haptic feedback via ESP32 for detected sounds
- Displays recognized sound type on screen

## Files in this repository
- `train.py` – Script to train the sound classification model  
- `live_serial_predict.py` – Script to run live predictions from ESP32  
- `label_map.json` – Maps sound categories to labels  
- `README.md` – Project overview

> **Note:** Audio dataset is not included in this repository.

## How to Run
1. Connect the ESP32 microphone module to your computer.  
2. Run `live_serial_predict.py` to start real-time sound recognition.  
3. The recognized sound and corresponding vibration are displayed and activated in real-time.

## Technology Stack
- **Python** for model training and predictions  
- **ESP32** for capturing audio and providing haptic feedback  
- **Machine Learning** for sound classification  



**Author:** Druhitha Duggirala  
**Contact:** druhithad@gmail.com  
