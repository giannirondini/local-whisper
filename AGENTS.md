# AGENTS.md - AI Agent Guidelines for LocalWhisper

> This document provides context and guidelines for AI agents, coding assistants, and LLMs working on this codebase.

## Project Summary

| Attribute | Value |
|-----------|-------|
| **Name** | LocalWhisper |
| **Type** | CLI Application |
| **Language** | Python 3.10+ |
| **Platform** | macOS |
| **Purpose** | Local voice-to-text with AI refinement |
| **Privacy** | 100% offline - no data leaves the machine |

## Repository Structure

```
LocalWhisper/
├── src/
│   └── localwhisper/
│       ├── __init__.py      # Package marker
│       ├── __main__.py      # Entry point
│       ├── cli.py           # CLI logic (formerly main.py)
│       ├── core.py          # AI logic (Whisper + Ollama)
│       └── audio.py         # Audio logic
├── tests/                   # Unit tests
├── requirements.txt # Python dependencies
├── README.md        # User documentation
├── AGENTS.md        # This file - AI agent guidelines
├── LICENSE          # MIT License
└── .github/
    └── copilot-instructions.md  # GitHub Copilot-specific instructions
```

## Core Components

### 1. AudioRecorder (src/localwhisper/audio.py)
- Captures microphone input using PyAudio
- Records in 16kHz mono WAV format
- Runs recording loop in a separate thread
- Saves to temporary files for processing

### 2. AIProcessor (src/localwhisper/core.py)
- **Transcription**: Uses `faster-whisper` with the `base.en` model
- **Refinement**: Calls local Ollama API (default: Mistral model)
- Handles both transcription and instruction-based modifications

### 3. LocalWhisperCLI (src/localwhisper/cli.py)
- Interactive terminal interface
- Global hotkey (`Cmd+Shift+G`) via `pynput`
- Manages recording state and user commands
- Copies final output to clipboard

## Development Guidelines

### DO ✅
- Keep all processing local (no external API calls except localhost Ollama)
- Use threading for blocking operations
- Handle exceptions gracefully with user-friendly messages
- Clean up temporary files after use
- Add type hints to new functions
- Follow existing emoji conventions for CLI output
- Test on macOS (primary target platform)

### DON'T ❌
- Add features requiring internet connectivity
- Store sensitive audio/text data persistently
- Block the main thread during recording or AI processing
- Remove or weaken existing error handling
- Add dependencies without updating `requirements.txt`

## Key Technical Details

### Audio Configuration
```python
SAMPLE_RATE = 16000  # Optimal for speech recognition
CHANNELS = 1         # Mono
CHUNK = 1024         # Buffer size
FORMAT = pyaudio.paInt16  # 16-bit PCM
```

### Ollama API Endpoint
```
POST http://localhost:11434/api/generate
Content-Type: application/json

{
  "model": "mistral",
  "prompt": "<text to refine>",
  "system": "<system prompt>",
  "stream": false
}
```

### Whisper Model Options
- `tiny.en` - Fastest, least accurate
- `base.en` - Good balance (current default)
- `small.en` - Better accuracy, slower
- `medium.en` - High accuracy, requires more RAM
- `large-v2` - Best accuracy, multilingual, slowest

## Common Modification Scenarios

| Task | File(s) to Modify |
|------|-------------------|
| Add new CLI command | `src/localwhisper/cli.py` - `print_menu()` and `run()` |
| Change transcription model | `src/localwhisper/core.py` - `AIProcessor.__init__()` |
| Modify refinement prompt | `src/localwhisper/core.py` - `refine_text()` |
| Adjust audio quality | `src/localwhisper/audio.py` - constants at top |
| Add new hotkey | `src/localwhisper/cli.py` - `HOTKEY_COMBINATION` and listener |

## Testing Checklist

Before committing changes, verify:
- [ ] Application starts without errors
- [ ] Hotkey triggers recording on/off
- [ ] Audio is captured and saved correctly
- [ ] Whisper transcription produces output
- [ ] Ollama refinement works (requires Ollama running)
- [ ] Clipboard copy succeeds
- [ ] Temporary files are cleaned up

## Dependencies

| Package | Purpose |
|---------|---------|
| `faster-whisper` | Local Whisper model inference |
| `pyaudio` | Audio capture |
| `pynput` | Global hotkey handling |
| `requests` | Ollama API communication |
| `pyperclip` | Clipboard operations |

## External Requirements

1. **Ollama**: Must be installed and running with a model pulled
   ```bash
   ollama pull mistral
   ```

2. **PortAudio**: Required for PyAudio on macOS
   ```bash
   brew install portaudio
   ```

## Contact & Contribution

This is a learning project. Contributions should maintain the privacy-first, offline-only philosophy of the application.
