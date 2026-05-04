# GitHub Copilot Instructions for LocalWhisper

## Project Overview

LocalWhisper is a privacy-focused, local voice-to-text application for macOS. It captures voice via a global hotkey, transcribes audio using OpenAI's Whisper model locally (via `faster-whisper`), and refines the transcription using a local LLM via Ollama.

## Architecture

The project follows a modular design with three main components:

- **src/localwhisper/cli.py**: CLI interface, hotkey handling, and orchestration
- **src/localwhisper/core.py**: AI processing (Whisper transcription + Ollama refinement)
- **src/localwhisper/audio.py**: Audio recording using PyAudio

## Code Style Guidelines

- Use Python 3.10+ features
- Follow PEP 8 conventions
- Use type hints for function parameters and return values
- Prefer explicit error handling with try/except blocks
- Use threading for non-blocking operations (audio recording, AI processing)
- Keep functions focused and single-purpose

## Key Dependencies

- `faster-whisper`: Local Whisper model for transcription
- `pyaudio`: Audio capture
- `pynput`: Global hotkey handling
- `requests`: Ollama API communication
- `pyperclip`: Clipboard operations

## Important Patterns

### Threading
Audio recording and AI processing run in separate threads to keep the CLI responsive:
```python
threading.Thread(target=self._process_audio, args=(wav_file,)).start()
```

### Ollama API
The project uses Ollama's REST API at `http://localhost:11434/api/generate`:
```python
data = {
    "model": self.ollama_model,
    "prompt": raw_text,
    "system": system_prompt,
    "stream": False
}
```

### Audio Configuration
Standard settings for voice recording:
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Chunk size: 1024
- Format: 16-bit PCM (WAV)

## When Generating Code

1. **Preserve privacy-first approach**: Never add features that send data externally
2. **Handle errors gracefully**: Always wrap I/O and API calls in try/except
3. **Clean up resources**: Delete temporary audio files after processing
4. **Provide user feedback**: Use emoji prefixes for status messages (🎤, ✅, ❌, etc.)
5. **Keep it offline**: All processing must work without internet connectivity
6. **macOS focus**: This project targets macOS; consider platform-specific behaviors. The global hotkey (`pynput.GlobalHotKeys`) requires the **Terminal** you are using (or your Python binary) to be listed under **System Settings → Privacy & Security → Accessibility**. Remind users to add `venv/bin/python` when documenting setup steps.

## Common Tasks

- Adding new CLI commands: Modify `LocalWhisperCLI` in `src/localwhisper/cli.py`
- Changing AI models: Update defaults in `AIProcessor.__init__()` in `src/localwhisper/core.py`
- Modifying audio settings: Update constants at the top of `src/localwhisper/audio.py`
- Adding new refinement prompts: Modify `refine_text()` in `src/localwhisper/core.py`
