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
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure you have Ollama running with a model:
   ```bash
   ollama pull mistral
   ```

## 📝 Making Changes

### Code Style
- Follow PEP 8 conventions
- Use type hints for function parameters and return values
- Add docstrings to new functions
- Use meaningful variable names

### Commit Messages
Use clear, descriptive commit messages:
- `feat: add support for multiple Whisper models`
- `fix: handle missing Ollama connection gracefully`
- `docs: update README with new features`
- `refactor: simplify audio recording logic`

### Testing Your Changes
Before submitting a PR:
- [ ] Application starts without errors
- [ ] Hotkey works correctly
- [ ] Recording and transcription function properly
- [ ] No regressions in existing features

## 🔧 Development Setup

### Prerequisites
- Python 3.10+
- PortAudio: `brew install portaudio`
- Ollama: [ollama.com](https://ollama.com)

### Running the App
```bash
python -m localwhisper
```

## 📋 Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Update documentation if needed
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
