---
description: Setup and Run EchoMind Wearable Memory Assistant Prototype
---

This workflow will set up the virtual environment, download the required offline models for Speech-to-Text and NLP, fetch flutter dependencies, and finally start both the Flutter frontend and Python backend.

## 1. Set up the backend environment and install models

// turbo
```powershell
cd c:\Project\EchoMind\EchoMind-Wearable-Memory-Assistant-main\backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile "vosk-model-small-en-us-0.15.zip"
Expand-Archive -Path "vosk-model-small-en-us-0.15.zip" -DestinationPath "." -Force
python -m spacy download en_core_web_sm
```

## 2. Set up the Flutter mobile app

// turbo
```powershell
cd c:\Project\EchoMind\EchoMind-Wearable-Memory-Assistant-main\mobile_app
flutter pub get
```

## 3. Run the application

// turbo
```powershell
cd c:\Project\EchoMind\EchoMind-Wearable-Memory-Assistant-main
.\run_all.bat
```
