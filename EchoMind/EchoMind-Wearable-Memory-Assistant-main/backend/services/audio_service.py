from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np


class AudioService:
    def __init__(self, vosk_model_path: str = "") -> None:
        # Ignore vosk_model_path, using industry-grade faster-whisper and VAD
        self._model = None
        self._vad_model = None
        self.get_speech_timestamps = None
        self._audio_buffer = {}  # session_id -> bytearray

    def _load_models(self) -> None:
        if self._model is not None:
            return
        
        from faster_whisper import WhisperModel
        # Faster Whisper multilingual base model for edge CPU performance
        self._model = WhisperModel("base", device="cpu", compute_type="int8")

        import torch
        # Load Silero VAD locally
        self._vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        (self.get_speech_timestamps, _, _, *_) = utils

    def transcribe_audio_chunk(self, session_id: str, audio_data: bytes) -> dict[str, Any] | None:
        self._load_models()
        import torch

        if session_id not in self._audio_buffer:
            self._audio_buffer[session_id] = bytearray()

        self._audio_buffer[session_id].extend(audio_data)

        # Evaluate dynamically every 0.5 seconds minimum (16000 bytes)
        if len(self._audio_buffer[session_id]) >= 16000:
            audio_np = np.frombuffer(self._audio_buffer[session_id], dtype=np.int16).astype(np.float32) / 32768.0
            
            # --- Proximity Energy Gating (Battery & Diarization Optimization) ---
            rms_energy = float(np.sqrt(np.mean(audio_np**2)))
            if rms_energy < 0.005:  # Ambient Drop Threshold
                self._audio_buffer[session_id].clear()
                return {"text": "", "final": False}
                
            # Volume-Proximity Heuristic
            speaker_tag = "Self" if rms_energy > 0.03 else "External"
            
            # Apply Silero VAD to evaluate current buffer
            tensor_audio = torch.from_numpy(audio_np)
            speech_timestamps = self.get_speech_timestamps(tensor_audio, self._vad_model, sampling_rate=16000)
            
            if not speech_timestamps:
                # No speech detected yet, keep buffer clear to save memory
                self._audio_buffer[session_id].clear()
                return {"text": "", "final": False}
            
            # Dynamic endpointing: detect if speaker paused
            last_speech_end = speech_timestamps[-1]['end']
            total_frames = len(audio_np)
            
            # 4800 frames = 0.3 seconds of silence signifies a pause
            has_paused = (total_frames - last_speech_end) > 4800
            # 80000 frames = 5 seconds max buffer size to prevent infinite hold
            hit_max_buffer = total_frames > 80000
            
            if has_paused or hit_max_buffer:
                segments, _ = self._model.transcribe(audio_np, beam_size=5, task="translate")
                text = " ".join([segment.text for segment in segments]).strip()
                
                if hit_max_buffer and not has_paused:
                    # If cut due to max time without a pause, leave 0.5 seconds overlap (16000 bytes)
                    overlap_bytes = self._audio_buffer[session_id][-16000:]
                    self._audio_buffer[session_id] = bytearray(overlap_bytes)
                else:
                    self._audio_buffer[session_id].clear()
                
                if text:
                    return {"text": text, "final": True, "speaker": speaker_tag}
        return None

    def finalize_session(self, session_id: str) -> dict[str, Any] | None:
        # Transcribe any remaining buffer on session close
        if session_id in self._audio_buffer and len(self._audio_buffer[session_id]) > 0:
            self._load_models()
            audio_np = np.frombuffer(self._audio_buffer[session_id], dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self._model.transcribe(audio_np, beam_size=5, task="translate")
            text = " ".join([segment.text for segment in segments]).strip()
            
            rms_energy = float(np.sqrt(np.mean(audio_np**2)))
            speaker_tag = "Self" if rms_energy > 0.03 else "External"
            
            del self._audio_buffer[session_id]
            if text:
                return {"text": text, "final": True, "speaker": speaker_tag}
        return None

    def transcribe_audio(self, audio_path: Path) -> dict[str, Any]:
        self._load_models()
        with wave.open(str(audio_path), "rb") as wf:
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                raise ValueError("Audio must be 16-bit mono PCM")
            audio_data = wf.readframes(wf.getnframes())

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, _ = self._model.transcribe(audio_np, beam_size=5, task="translate")
        text = " ".join([segment.text for segment in segments]).strip()
        
        rms_energy = float(np.sqrt(np.mean(audio_np**2)))
        speaker_tag = "Self" if rms_energy > 0.03 else "External"
        
        return {"text": text, "chunks": [{"text": text, "final": True, "speaker": speaker_tag}], "speaker": speaker_tag}
