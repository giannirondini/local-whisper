"""Audio recording module for LocalWhisper.

This module handles microphone input capture using PyAudio,
recording audio in a format suitable for Whisper transcription.
"""

from __future__ import annotations

import pyaudio
import wave
import tempfile
import threading
from typing import Optional

# Configuration Constants
SAMPLE_RATE: int = 16000
CHANNELS: int = 1
CHUNK: int = 1024


class AudioRecorder:
    """Records audio from the microphone and saves to temporary WAV files."""

    def __init__(self) -> None:
        """Initialize the audio recorder."""
        self.recording: bool = False
        self.frames: list[bytes] = []
        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

    def start_recording(self) -> None:
        """Start a new recording session."""
        self.recording = True
        self.frames = []
        
        try:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=CHANNELS,
                                      rate=SAMPLE_RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
            print("🎤 Recording started...")
            threading.Thread(target=self._record_loop).start()
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            self.recording = False

    def _record_loop(self) -> None:
        """Internal loop to read audio frames."""
        while self.recording:
            try:
                data = self.stream.read(CHUNK)
                self.frames.append(data)
            except Exception as e:
                print(f"Error recording: {e}")
                break

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save to a temporary WAV file.

        Returns:
            Path to the temporary WAV file, or None if no audio was recorded.
        """
        self.recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        print("🛑 Recording stopped.")
        return self._save_temp_wav()

    def _save_temp_wav(self) -> Optional[str]:
        """Save recorded frames to a temporary file.

        Returns:
            Path to the temporary WAV file, or None if no frames were recorded.
        """
        if not self.frames:
            return None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_filename = f.name
            
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return temp_filename
        except Exception as e:
            print(f"❌ Error saving audio file: {e}")
            return None
