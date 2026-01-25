"""AI processing module for LocalWhisper.

This module handles speech-to-text transcription using Whisper
and text refinement using a local Ollama LLM.
"""

from __future__ import annotations

import time
from typing import Optional

import requests
from faster_whisper import WhisperModel


class AIProcessor:
    """Handles AI-powered transcription and text refinement."""

    def __init__(self, whisper_size: str = "base.en", ollama_model: str = "mistral") -> None:
        """Initialize the AI processor.

        Args:
            whisper_size: Whisper model size (e.g., 'tiny.en', 'base.en', 'small.en').
            ollama_model: Ollama model name for text refinement.
        """
        self.whisper_size: str = whisper_size
        self.ollama_model: str = ollama_model
        
        print(f"🎧 Loading Whisper model: {self.whisper_size}...")
        try:
            # On Apple Silicon, faster-whisper runs on CPU by default.
            self.model = WhisperModel(self.whisper_size, device="cpu", compute_type="int8")
            print("✅ Whisper model loaded.")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            raise

    def transcribe(self, file_path: str) -> Optional[str]:
        """Transcribe audio file to text using Whisper.

        Args:
            file_path: Path to the audio file to transcribe.

        Returns:
            Transcribed text, or None if transcription failed.
        """
        try:
            print(f"Analyzing audio: {file_path}...")
            start = time.time()
            
            # beam_size=5 is standard for accuracy
            segments, info = self.model.transcribe(file_path, beam_size=5)
            
            # Combine all segments into one string
            transcribed_text = " ".join([segment.text for segment in segments])
            
            duration = time.time() - start
            print(f"✅ Transcription complete in {duration:.2f}s")
            return transcribed_text.strip()
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return None

    def refine_text(self, raw_text: str, instruction: Optional[str] = None) -> str:
        """Refine text using a local Ollama LLM.

        Args:
            raw_text: The raw transcribed text to refine.
            instruction: Optional instruction for how to modify the text.

        Returns:
            Refined text, or an error message if refinement failed.
        """
        print(f"🧠 Sending to Ollama ({self.ollama_model})...")
        
        system_prompt = (
            "You are a helpful assistant designed to transcribe voice notes. "
            "Your task is to correct grammar, punctuation, and capitalization "
            "of the following text. Do not add any conversational filler. "
            "Output ONLY the corrected text."
        )

        if instruction:
            system_prompt += f"\n\nUSER INSTRUCTION: {instruction}\nFollow this instruction strictly."
        
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.ollama_model,
            "prompt": raw_text,
            "system": system_prompt,
            "stream": False 
        }
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()['response'].strip()
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {e}"
        except (KeyError, ValueError) as e:
             return f"Error parsing Ollama response: {e}"
