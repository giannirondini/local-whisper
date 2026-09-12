# AGENTS.md

The single reference for AI agents working on this repository is
**[.github/copilot-instructions.md](.github/copilot-instructions.md)**. Read it before changing
anything; it covers the architecture, threading rules, Ollama payload, configuration layer,
development workflow and the testing checklist. `CLAUDE.md` points to the same file.

Non-negotiables, in case you read nothing else:

- Everything stays local: the only network calls are Ollama on `localhost` and the one-time Whisper model download.
- Only `src/localwhisper/cli.py` may `print()`; the engine reports through events.
- All AI work runs as jobs on the engine's single worker thread; never call `AIProcessor` from a presenter.
- Dependencies live in `pyproject.toml` only; use `uv add`, commit `uv.lock`.
- Before finishing: `uv run ruff check && uv run ruff format --check && uv run mypy src && uv run pytest -q`.
