"""Audio recording module for LocalWhisper.

This module captures microphone input with PyAudio in callback mode and returns
the recording as an in-memory float32 buffer ready for Whisper. No audio is
written to disk unless `save_wav()` is called explicitly.
"""

from __future__ import annotations

import logging
import threading
import wave
from collections.abc import Mapping

import numpy as np
import pyaudio

# Configuration Constants
SAMPLE_RATE: int = 16000
CHANNELS: int = 1
CHUNK: int = 1024
SAMPLE_FORMAT: int = pyaudio.paInt16
_INT16_SCALE: float = 32768.0

logger = logging.getLogger(__name__)


class AudioDeviceError(Exception):
    """Raised when the input stream cannot be opened (no device, permission denied...)."""


class AudioRecorder:
    """Records audio from the microphone into memory.

    PortAudio invokes `_callback` on its own thread for every CHUNK of samples;
    the recorder only appends the raw PCM bytes. This removes the Python read loop
    and the overflow/race problems that came with it.
    """

    def __init__(self) -> None:
        """Initialize the audio recorder."""
        self.recording: bool = False
        self.frames: list[bytes] = []
        self.dropped_chunks: int = 0
        self.p: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream: pyaudio.Stream | None = None
        self._frames_lock = threading.Lock()
        self._terminated = False

    def start_recording(self) -> None:
        """Start a new recording session.

        Raises:
            AudioDeviceError: If the input stream cannot be opened. The recorder is
                left idle so a later attempt can succeed.
        """
        if self._terminated:
            raise AudioDeviceError("Recorder has been terminated")
        with self._frames_lock:
            self.frames = []
            self.dropped_chunks = 0

        try:
            self.stream = self.p.open(
                format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self._callback,
            )
            self.recording = True
            logger.info("Recording started")
        except OSError as e:
            self.stream = None
            self.recording = False
            raise AudioDeviceError(str(e)) from e

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: Mapping[str, float],
        status: int,
    ) -> tuple[bytes | None, int]:
        """PortAudio callback: store the incoming PCM chunk.

        Runs on PortAudio's thread. Must stay cheap and must not raise.
        """
        if status & pyaudio.paInputOverflow:
            self.dropped_chunks += 1
        if in_data:
            with self._frames_lock:
                self.frames.append(in_data)
        return (None, pyaudio.paContinue)

    def stop_recording(self) -> np.ndarray | None:
        """Stop recording and return the captured audio.

        Returns:
            Mono float32 samples in [-1, 1] at SAMPLE_RATE, or None if nothing was
            captured.
        """
        self.recording = False
        if self.stream is not None:
            try:
                self.stream.stop_stream()  # blocks until the last callback returns
                self.stream.close()
            except OSError as e:
                logger.warning("Error closing audio stream: %s", e)
            self.stream = None

        logger.info("Recording stopped (%d chunk(s) dropped)", self.dropped_chunks)

        with self._frames_lock:
            pcm = b"".join(self.frames)
            self.frames = []

        if not pcm:
            return None
        return pcm_to_float32(pcm)

    def duration_seconds(self) -> float:
        """Seconds of audio buffered so far."""
        with self._frames_lock:
            samples = sum(len(chunk) for chunk in self.frames) // 2  # 2 bytes per int16
        return samples / SAMPLE_RATE

    def terminate(self) -> None:
        """Release PortAudio. Idempotent; the recorder is unusable afterwards."""
        if self._terminated:
            return
        self._terminated = True
        if self.stream is not None:
            self.stop_recording()
        self.p.terminate()


def pcm_to_float32(pcm: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to a float32 array in [-1, 1]."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / _INT16_SCALE


def save_wav(audio: np.ndarray, path: str) -> None:
    """Write a float32 buffer to a 16-bit mono WAV file. Debugging aid only."""
    pcm = np.clip(audio * _INT16_SCALE, -_INT16_SCALE, _INT16_SCALE - 1).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
