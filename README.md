# LocalWhisper 🎙️🤖

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)

**LocalWhisper** is a privacy-focused, local voice-to-text tool for macOS. It captures your voice with a global hotkey, transcribes it using OpenAI's **Whisper** model locally, and refines the text (grammar, punctuation) using a local LLM via **Ollama**.

> **Note**: This project is for educational purposes and runs completely offline. No audio or text leaves your machine.

## ✨ Features

- **Global Hotkey**: Press `Cmd+Shift+G` anywhere to start/stop recording.
- **Local Transcription**: Uses `faster-whisper` for high-performance offline speech-to-text.
- **AI Polishing**: Uses `Ollama` (Mistral/Llama3) to fix grammar, remove filler words like "um/uh", and format text.
- **Interactive CLI**:
    - Record new notes.
    - Modify the last note with voice instructions (e.g., "Make it a bulleted list").
    - Modify with text instructions.
- **Clipboard Injection**: Automatically copies the final text to your clipboard.

## 🛠️ Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com) and pull a model:
    ```bash
    ollama pull mistral
    ```
3.  **PortAudio**: Required for microphone access.
    ```bash
    brew install portaudio
    ```
4.  **Accessibility Permission**: The global hotkey requires the **Terminal you are using** itself to be trusted. Go to **System Settings → Privacy & Security → Accessibility** and add your terminal application (*e.g., Terminal, iTerm2, Ghostty*).

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/gianni/LocalWhisper.git
    cd LocalWhisper
    ```

2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    # To install in editable mode:
    pip install -e .
    ```

## 🎮 Usage

1.  Start the application:
    ```bash
    # Run as a module (recommended)
    python -m localwhisper

    # Or if installed in editable mode:
    # localwhisper
    ```

2.  **Record**:
    - Press **`Cmd+Shift+G`** to start recording.
    - Speak your thought.
    - Press **`Cmd+Shift+G`** again to stop.
    - The tool will transcribe, refine, and copy the text to your clipboard!

3.  **Interactive Mode**:
    The terminal window provides additional options:
    - `[v]`: **Voice Modify**. Press `v`, then use the hotkey to record an instruction (e.g., "Translate to Spanish").
    - `[m]`: **Text Modify**. Type an instruction to change the last captured text.

## 🧠 Under the Hood

- **Ears**: `faster-whisper` (default: `base.en` model).
- **Brain**: `Ollama` (default: `mistral` model).
- **Body**: Python `pynput` for hotkeys and `pyaudio` for recording.

## 📄 License

MIT License.
