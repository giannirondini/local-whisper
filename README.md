# LocalWhisper 🎙️🤖

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)

**LocalWhisper** is a privacy-focused, local voice-to-text tool for macOS. It captures your voice with a global hotkey, transcribes it using OpenAI's **Whisper** model locally, and refines the text (grammar, punctuation) using a local LLM via **Ollama**.

> **Note**: This project is for educational purposes. No audio or text leaves your machine: the only network calls are to Ollama on `localhost` and a one-time download of the Whisper model from Hugging Face on first run (about 150 MB for `base.en`, cached in `~/.cache/huggingface`). After that it runs fully offline.

## ✨ Features

- **Global Hotkey**: Press `Cmd+Shift+G` anywhere to start/stop recording.
- **Local Transcription**: Uses `faster-whisper` for high-performance offline speech-to-text.
- **AI Polishing**: Uses `Ollama` (default model `gemma4:e2b-mlx`, see `docs/OLLAMA_MODEL_DECISION.md`) to fix grammar, remove filler words like "um/uh", and format text.
- **Interactive CLI**:
    - Record new notes.
    - Modify the last note with voice instructions (e.g., "Make it a bulleted list").
    - Modify with text instructions. Edits compose: each instruction applies to the current text, and `[u]` undoes the last one.
- **Menu Bar App** (optional): a status item that shows whether you are idle, recording or processing, notifications when text is ready, voice/text editing, undo, a Settings submenu and optional auto-paste. Same hotkey, same engine.
- **Configurable**: model names, language, Ollama URL and hotkey via a config file, environment variables or flags.
- **Clipboard Injection**: Automatically copies the final text to your clipboard.

## 🛠️ Prerequisites

1.  **Python 3.12+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com) and pull a model:
    ```bash
    ollama pull gemma4:e2b-mlx
    ```
3.  **PortAudio**: Required for microphone access.
    ```bash
    brew install portaudio
    ```
4.  **macOS permissions**: the global hotkey requires the **terminal you are using** (or the Python binary, e.g. `.venv/bin/python`) to be listed under **System Settings → Privacy & Security → Accessibility** *and* under **Input Monitoring** (recent macOS versions need both). The first recording will also prompt for **Microphone** access.

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/gianni/LocalWhisper.git
    cd LocalWhisper
    ```

2.  Install with [uv](https://docs.astral.sh/uv/) (recommended; pins exact versions from `uv.lock`):
    ```bash
    uv sync                 # CLI only
    uv sync --all-extras    # CLI + menu bar app (adds rumps and PyObjC)
    ```
    A later plain `uv sync` performs an exact sync and removes rumps/PyObjC again if they were
    installed this way; re-run with `--all-extras` (or `--extra gui`) to bring them back.
    Or with pip into a virtualenv of your own:
    ```bash
    pip install -e .          # or: pip install -e ".[gui]"
    ```

## 🎮 Usage

1.  Start the application:
    ```bash
    uv run localwhisper          # or: uv run python -m localwhisper
    localwhisper --help          # all flags; see Configuration below
    ```
    (With a pip install, just `localwhisper` or `python -m localwhisper`.)

2.  **Record**:
    - Press **`Cmd+Shift+G`** to start recording.
    - Speak your thought.
    - Press **`Cmd+Shift+G`** again (or ENTER in the terminal) to stop.
    - The tool will transcribe, refine, and copy the text to your clipboard!

3.  **Interactive Mode**:
    The terminal window provides additional options:
    - `[v]`: **Voice Modify**. Press `v`, speak an instruction (e.g., "Translate to Spanish"), press ENTER to stop.
    - `[m]`: **Text Modify**. Type an instruction to change the current text.
    - `[u]`: **Undo** the last edit. Undoing everything gives you the raw transcript back.
    - `[s]`: **Show** the current text and the instructions applied so far.

4.  **Menu Bar App** (requires the `gui` extra):
    ```bash
    uv run localwhisper-gui
    ```
    A status item appears in the menu bar: ⏳ while the Whisper model loads, 🎙 when ready, 🔴 while recording, 🟠 while recording a spoken instruction, 📝/🧠 while transcribing/refining. The hotkey works exactly as in the CLI. The menu offers:
    - **Record** / **Stop Recording** (same as the hotkey).
    - **Modify with Voice** / **Stop Instruction**, **Modify with Text…**, **Undo Last Edit**: the `[v]`, `[m]` and `[u]` commands of the CLI.
    - **Show Last Text…** (with a Copy button) and **Copy Again**.
    - A status line with the last result or error.
    - **Settings ▸**: Whisper model, Ollama model, language and hotkey as text fields, plus an **Auto-paste** checkbox. Changes are written to the config file (other lines and comments are kept). Ollama model, language and hotkey apply immediately; the Whisper model applies on the next launch.
    - **Quit LocalWhisper**.

    Every result is copied to the clipboard and announced with a macOS notification. Until the app is packaged as an `.app` (see `docs/UI_PLAN.md`), notifications are posted through `osascript`, so macOS attributes them to *Script Editor*; if you see none, check **System Settings → Notifications → Script Editor**. With **Auto-paste** on, the app also presses `Cmd+V` in the frontmost application right after copying (this needs the same Accessibility permission as the hotkey). It accepts the same flags, config file and environment variables as the CLI.

## ⚙️ Configuration

Precedence, lowest to highest: defaults → `~/.config/localwhisper/config.toml` → `LOCALWHISPER_*` environment variables → command-line flags (`localwhisper --help`).

```toml
# ~/.config/localwhisper/config.toml
whisper_model = "small"      # base.en (default, English-only), base, small, medium, large-v3
language = "it"              # omit for auto-detect; *.en models accept only "en"
ollama_model = "gemma4:e2b-mlx"
ollama_url = "http://localhost:11434"
ollama_timeout = 120         # seconds to wait for one refinement
keep_alive = "5m"           # how long Ollama keeps the model loaded
hotkey = "<cmd>+<shift>+g"   # pynput syntax
auto_paste = false           # menu bar app: press Cmd+V after copying a result
```

The same keys work as `LOCALWHISPER_WHISPER_MODEL`, `LOCALWHISPER_LANGUAGE`, ... or as `--whisper-model`, `--language`, .... Note that `Cmd+Shift+G` is also "find previous" in most browsers; pick another combination with `hotkey` if that bites.

## 🧠 Under the Hood

- **Ears**: `faster-whisper` (default: `base.en` model).
- **Brain**: `Ollama` (default: `gemma4:e2b-mlx`, called with `think: false`; the transcript is sent as delimited content so the model edits it instead of answering it).
- **Body**: Python `pynput` for hotkeys and `pyaudio` for recording.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [.github/copilot-instructions.md](.github/copilot-instructions.md) for the architecture, threading rules and testing checklist. Design notes live in [docs/](docs/).

## 📄 License

MIT License.
