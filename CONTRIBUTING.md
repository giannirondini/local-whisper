# Contributing to LocalWhisper

Thank you for your interest in contributing to LocalWhisper! This document provides guidelines for contributing to the project.

## 🎯 Project Philosophy

LocalWhisper is built on these core principles:

1. **Privacy First**: All processing happens locally. Never add features that send data externally.
2. **Simplicity**: Keep the codebase small and understandable.
3. **Offline Operation**: The app must work without internet (Ollama runs locally).

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LocalWhisper.git
   cd LocalWhisper
   ```
3. Install dependencies (runtime + dev tools, pinned by `uv.lock`):
   ```bash
   brew install portaudio
   uv sync --all-extras    # drop --all-extras if you only work on the CLI
   ```
   A later plain `uv sync` (no flags) performs an exact sync and removes rumps/PyObjC again since
   they are an optional extra; always pass `--all-extras` while touching `src/localwhisper/gui/`.
4. Make sure you have Ollama running with the default model:
   ```bash
   ollama pull gemma4:e2b-mlx
   ```
5. Read [.github/copilot-instructions.md](.github/copilot-instructions.md): it is the reference for the architecture, the threading rules and where things go.

## 📝 Making Changes

### Code Style
- `ruff` enforces PEP 8 and import order; `ruff format` formats (no black)
- Type hints on every function signature; `mypy src` must stay clean
- Add docstrings to new public functions
- Dependencies go in `pyproject.toml` via `uv add`; commit the updated `uv.lock`. There is no `requirements.txt`.

### Commit Messages
Use clear, descriptive commit messages:
- `feat: add support for multiple Whisper models`
- `fix: handle missing Ollama connection gracefully`
- `docs: update README with new features`
- `refactor: simplify audio recording logic`

### Testing Your Changes
CI runs these on macOS for Python 3.12; run them locally first:
```bash
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest -q
```
The manual checklist (hotkey, silent recording, Ollama down, clean exit) is in [.github/copilot-instructions.md](.github/copilot-instructions.md#testing-checklist).

## 🔧 Development Setup

### Prerequisites
- Python 3.12+ (`uv sync` picks up `.python-version`; older versions lack `onnxruntime` wheels, a faster-whisper dependency)
- [uv](https://docs.astral.sh/uv/)
- PortAudio: `brew install portaudio`
- Ollama: [ollama.com](https://ollama.com)
- **macOS permissions**: add your terminal (or `.venv/bin/python`) under **System Settings → Privacy & Security → Accessibility** and **Input Monitoring** for the CLI's global hotkey (the menu bar app's native hotkey needs neither); allow **Microphone** on first recording

### Running the App
```bash
uv run localwhisper            # or: uv run python -m localwhisper
uv run localwhisper -v         # engine logs on stderr
uv run localwhisper-gui        # menu bar app (needs the `gui` extra)
packaging/build_app.sh         # dist/LocalWhisper.app via PyInstaller (the `package` group)
```
Never add a test that builds the bundle; `build/` and `dist/` are git-ignored.

## 📋 Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Update documentation in the same PR: `.github/copilot-instructions.md` for architecture/patterns, `README.md` for user-visible behaviour
4. Submit a pull request with a clear description

## 🐛 Reporting Issues

When reporting bugs, please include:
- macOS version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Any error messages

## 💡 Feature Requests

Feature ideas are welcome! Please open an issue with:
- Clear description of the feature
- Use case / motivation
- Any implementation ideas

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
