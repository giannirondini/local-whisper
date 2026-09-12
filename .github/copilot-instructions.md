# LocalWhisper — Reference for Coding Agents

This file is the single source of truth for AI coding assistants (Copilot, Claude Code, Codex, ...)
working on this repository. `AGENTS.md` and `CLAUDE.md` only point here. Human-facing docs live in
`README.md` (usage) and `CONTRIBUTING.md` (workflow); design history lives in `docs/`.

## Project Overview

LocalWhisper is a privacy-focused, local voice-to-text application for macOS. It captures voice via a
global hotkey, transcribes audio using OpenAI's Whisper model locally (via `faster-whisper`), refines
the transcription using a local LLM via Ollama, and copies the result to the clipboard.

| Attribute | Value |
|-----------|-------|
| Language | Python 3.12+ (see `.python-version`) |
| Platform | macOS (PyAudio, `pynput` hotkeys, clipboard) |
| Privacy | All processing local. The only network calls are to Ollama on `localhost` and the one-time Whisper model download from Hugging Face. |
| Tooling | `uv` for environments and locking, `ruff` for lint/format, `mypy` for types, `pytest` for tests, GitHub Actions for CI |

## Repository Layout

```
LocalWhisper/
├── src/localwhisper/
│   ├── __init__.py      # package version
│   ├── __main__.py      # `python -m localwhisper` → cli.main()
│   ├── cli.py           # terminal presenter: menu, input(), emoji output, clipboard, hotkey binding
│   ├── engine.py        # headless engine: State machine, one worker thread, revision history, shutdown()
│   ├── events.py        # State / RecordingKind enums and the frozen Event dataclasses
│   ├── core.py          # AIProcessor: Whisper transcription + Ollama refinement + startup probe
│   ├── audio.py         # AudioRecorder: PyAudio callback mode → in-memory float32 buffer
│   ├── config.py        # Settings dataclass + load_settings() (file / env / flags)
│   ├── prompts.py       # Ollama system prompts, prompt builders, sampling options
│   └── gui/             # menu bar app (optional extra `gui`; `localwhisper-gui`)
│       ├── presenter.py # toolkit-free: events → MenuModel, menu clicks → engine calls, settings edits
│       ├── menubar.py   # rumps shell: status item, menu, dialogs, main-thread timer, hotkey, main()
│       └── notify.py    # user notifications via osascript (no bundle id outside an .app)
├── tests/               # pytest; one file per module, fakes instead of real audio/models
├── docs/
│   ├── CODE_REVIEW.md           # findings F-01..F-26 and the S1..S5 remediation log
│   ├── UI_PLAN.md               # menu bar GUI plan (depends on the engine being headless)
│   └── OLLAMA_MODEL_DECISION.md # why the default model is gemma4:e2b-mlx
├── .github/workflows/ci.yml     # macOS, Python 3.12: ruff, mypy, pytest
├── pyproject.toml       # single source of dependencies, dev group, tool config
└── uv.lock              # pinned versions; commit it
```

## Architecture

The project is a headless engine with presenters on top. The CLI is the first presenter; the menu
bar app in `gui/` is the second one (`docs/UI_PLAN.md`; phase 1 is complete).

- **`engine.py`** — `Engine` owns the recorder, the AI processor, one `RLock`, one `State`, one worker
  thread and the current note's revision history. Presenters call `start(kind)`, `stop()`, `toggle()`,
  `modify(instruction)`, `undo()`, `shutdown()` and receive events.
- **`events.py`** — `State` (`IDLE`, `RECORDING_NOTE`, `RECORDING_INSTRUCTION`, `PROCESSING`),
  `RecordingKind` (`NOTE`, `INSTRUCTION`) and the events: `StateChanged`, `RecordingStarted`,
  `RecordingStopped`, `Transcribing`, `TranscriptReady`, `Refining`, `RefinementReady`, `Reverted`,
  `Notice`, `Error`.
- **`cli.py`** — `LocalWhisperCLI`: menu loop over `input()`, renders events with emoji prefixes, copies
  `RefinementReady`/`Reverted` text to the clipboard, binds the global hotkey to `engine.toggle()`.
- **`gui/presenter.py`** — `MenuBarPresenter`: `post(event)` queues from any thread; `drain()` on the
  main thread folds queued events into an immutable `MenuModel` and returns it when it differs from
  the last one returned. The model stores facts (`icon`, `status`, `state`, `has_text`, `settings`);
  item titles and enabled flags are derived properties. `attach(engine, warning)` / `fail(message)`
  carry the outcome of the background model load through the same queue. Commands (`toggle_record`,
  `toggle_instruction`, `modify`, `undo`, `copy_again`, `update_setting`) run on the main thread and
  may change the model directly. Side effects are injected callables: `copy`, `paste` (auto-paste),
  `notify`, `save` (config file), `rebind_hotkey`. Every delivery (`RefinementReady`, `Reverted`)
  copies, notifies, and pastes if `settings.auto_paste`. No `rumps` import.
- **`gui/menubar.py`** — `MenuBarApp(rumps.App)`: builds the menu (Record, Modify with Voice, Modify
  with Text…, Undo, Show Last Text…, Copy Again, status line, Settings ▸, Quit), drains the presenter
  from a `rumps.Timer` (0.1 s), applies the model, opens `rumps.Window` dialogs, and runs `on_quit`
  before `rumps.quit_application()`. `HotkeyBinding` owns the pynput listener so Settings can rebind
  it. `main()` loads the engine on a `localwhisper-loader` thread and shuts down in a `finally` so
  Ctrl+C from a terminal is clean too. Every modal (`rumps.Window` or `rumps.alert`) is shown through
  `MenuBarApp._activate_and_run()`, never directly: clicking a status-bar item does not make the
  process frontmost, so a modal shown directly can be visible while keystrokes still go to whichever
  app *was* frontmost, typing silently does nothing (github.com/jaredks/rumps/issues/127, reproduced
  in this app). `_activate_and_run` reactivates first and hands focus back afterwards; skipping it for
  a new dialog reintroduces the bug for that one dialog only.
- **`gui/notify.py`** — `notify(title, message)` through `osascript` (non-blocking `Popen`, text as
  arguments). `UNUserNotificationCenter` needs a bundle identifier, which only the U4 `.app` has. A
  notification can be silently dropped by the OS, so it is not the only channel for a failure: see
  `alert` below.
- **`core.py`** — `AIProcessor(settings)`: `transcribe(audio) -> str` (empty string means no speech),
  `refine_text(text, instruction=None) -> str`, `check_ollama() -> str | None`. Raises
  `TranscriptionError` / `RefinementError`; never returns error strings.
- **`audio.py`** — `AudioRecorder`: `start_recording()` (raises `AudioDeviceError`), `stop_recording()
  -> np.ndarray | None`, `terminate()`. 16 kHz mono float32 in `[-1, 1]`, never written to disk except
  via the explicit `save_wav()` debugging helper.
- **`config.py`** — frozen `Settings` dataclass, validated in `__post_init__`; `load_settings(argv)`.
- **`prompts.py`** — `REFINE_SYSTEM_PROMPT`, `EDIT_SYSTEM_PROMPT`, `build_refine_prompt()`,
  `build_edit_prompt()`, `OLLAMA_OPTIONS`.

**Hard rules**
- Only `cli.py` may `print()`. Every other module uses `logging.getLogger(__name__)` and reports to
  the user through events. The one exception is `gui/menubar.py::main()` writing a configuration
  error to stderr before any UI exists.
- The engine never imports `pyperclip`, `pynput`, or any UI toolkit. Clipboard and hotkey are
  presenter concerns.
- Only `gui/menubar.py` imports `rumps`. Anything that maps events to what the menu shows belongs in
  `gui/presenter.py`, where it is testable without the `gui` extra.
- Runtime settings changes go through `Engine.update_settings()`; presenters never touch
  `AIProcessor`. Only Ollama/language settings apply live; `whisper_model` needs a restart.
- `Engine._apply_instruction()` emits only `Error` on a refine failure, no fallback `RefinementReady`
  (unlike a plain note, which falls back to the raw transcript): the current text is simply kept. In
  a background app that is invisible unless something loud says so, which is what `alert` (the
  presenter's blocking-dialog callable, `MenuBarApp.show_alert`) is for. Any new failure path that can
  end with nothing on the clipboard needs the same treatment, not just a status line or notification.
- `last_raw_text` / `last_refined_text` are replaced only once a non-empty result exists. An empty,
  too-short or failed recording never wipes the previous note.

### Runtime flow

1. Hotkey or `[r]` → `engine.toggle()` → `AudioRecorder.start_recording()` → `State.RECORDING_NOTE`.
2. Hotkey, ENTER or `[r]` again → `engine.stop()` → audio buffer queued as a job → `State.PROCESSING`.
3. Worker: `transcribe()` → `TranscriptReady` → `refine_text()` → `RefinementReady` (or `Error` +
   `RefinementReady(fallback_raw=True)` with the raw transcript if Ollama fails).
4. Presenter copies the text to the clipboard. `State.IDLE` once the queue is empty.
5. `[v]` records a spoken instruction (`RecordingKind.INSTRUCTION`; the hotkey is ignored while it
   runs), `[m]` types one. Both are jobs on the same worker and apply to the **current** text, appending
   a `Revision(instruction, text)`. `[u]` pops one revision; undoing everything yields the raw transcript.

### Threading and state

- One worker thread runs all AI jobs in FIFO order. Never spawn ad-hoc threads for processing and
  never call `AIProcessor` from a presenter.
- Every state change goes through `Engine._transition()` under the engine lock. New rules belong in
  the transition table, not as boolean flags on the presenter.
- A recording may start while `PROCESSING`; its job queues behind the running one. An instruction
  queued behind an in-flight note is applied to that note once it exists.
- Events are delivered synchronously, inside the engine lock, on the thread that produced them
  (worker or hotkey thread). Presenter callbacks must be quick and must not call back into the
  engine. The GUI marshals to the main thread through `MenuBarPresenter.post()` → `drain()`; never
  touch a `rumps`/AppKit object from an engine callback.
- Exit is a normal return from `main()` after `engine.shutdown()`, which stops any recording, drops
  pending jobs, joins the running one with a timeout and calls `AudioRecorder.terminate()`. Never use
  `os._exit()`.

### Ollama API

`AIProcessor.refine_text()` calls `POST {ollama_url}/api/generate`. The transcript is always wrapped
in `<<< >>>` delimiters and the model is told it is content to edit, not a message to answer
(review F-07). The payload always includes:

```python
{
    "model": settings.ollama_model,   # default gemma4:e2b-mlx, see docs/OLLAMA_MODEL_DECISION.md
    "prompt": prompts.build_refine_prompt(text),   # or build_edit_prompt(text, instruction)
    "system": prompts.REFINE_SYSTEM_PROMPT,        # or EDIT_SYSTEM_PROMPT
    "stream": False,
    "think": False,        # mandatory: thinking models return "" otherwise
    "keep_alive": settings.keep_alive,
    "options": prompts.OLLAMA_OPTIONS,   # temperature 0.2, num_ctx 4096, num_predict 1024
}
```

An empty response raises `RefinementError`; the engine then falls back to the raw transcript.
`check_ollama()` probes `GET /api/tags` once at startup (3 s cap) and returns a warning string when
Ollama is down or the configured model is not pulled; it never raises.

### Configuration

`config.py` defines `Settings` and `load_settings()`. Precedence, lowest to highest: dataclass
defaults → `~/.config/localwhisper/config.toml` (or `$XDG_CONFIG_HOME/localwhisper/config.toml`,
or `$LOCALWHISPER_CONFIG`) → `LOCALWHISPER_<FIELD>` environment variables → CLI flags. Fields:
`whisper_model`, `language`, `ollama_model`, `ollama_url`, `ollama_timeout`, `keep_alive`, `hotkey`,
`auto_paste`, `verbose`. Never add a new tunable as a module constant: add a `Settings` field with a
docstring and validation, and it becomes readable from all three sources and `--help` automatically.
`save_setting(name, value, path)` writes one key back to the config file in place (other lines and
comments kept; `None` removes the key); the menu bar Settings submenu uses it.

### Audio

16 000 Hz, mono, 1024-frame chunks, 16-bit PCM from PortAudio, converted to float32 in memory.
Recordings shorter than 0.5 s are dropped before Whisper; Whisper runs with VAD on and
`condition_on_previous_text=False` to limit hallucinations on silence.

## Development Workflow

```bash
brew install portaudio          # PyAudio build dependency
uv sync --all-extras             # runtime + dev deps + rumps/PyObjC for the menu bar app
uv run localwhisper --help       # or: uv run python -m localwhisper
uv run localwhisper-gui          # menu bar app; same flags, config file and env vars
uv run pytest -q                # ~130 tests, < 1 s, no microphone or model needed
uv run ruff check && uv run ruff format --check
uv run mypy src
```

- `pyproject.toml` is the only place dependencies are declared (`[project.dependencies]` for runtime,
  `[dependency-groups] dev` for tooling). There is no `requirements.txt`. Add a dependency with
  `uv add <pkg>` (or `uv add --group dev <pkg>`) so `uv.lock` is updated in the same change.
- `pip` users: `pip install -e . --group dev` (pip ≥ 25.1).
- `uv sync` alone performs an exact sync and drops anything not declared for that command, so a
  plain `uv sync` after `uv sync --all-extras` silently uninstalls rumps/PyObjC again (uv has no
  "always install this extra" setting). Always pass `--all-extras` while touching `gui/`.
- CI (`.github/workflows/ci.yml`) runs the four commands above on `macos-latest` for Python 3.12
  with `uv sync --frozen --all-extras`. A PR is green only if the lock file matches
  `pyproject.toml`. The `rumps` shell tests in `tests/test_gui.py` skip when the extra is missing.
- Tests replace the recorder and the AI with fakes (`tests/test_engine.py`) or `MagicMock`; the engine
  worker thread is real. Never make a test depend on a microphone, a model download or Ollama.

## Code Style

- Python 3.12+; `from __future__ import annotations`; `X | None` over `Optional[X]`.
- PEP 8 via `ruff` (rules `E, F, I, UP, B, BLE, SIM, RUF`, line length 100). `ruff format` replaces black.
- Type hints on every function signature; `mypy src` must stay clean.
- Catch specific exceptions and re-raise the module's typed error (`AudioDeviceError`,
  `TranscriptionError`, `RefinementError`). A blind `except Exception` is acceptable only as the last
  resort in the worker loop or an event callback, and must log with `logger.exception`.
- Docstrings on public functions; explain *why* in comments, not *what*.

## When Generating Code

1. **Privacy first**: never add features that send data anywhere but `localhost` Ollama.
2. **Keep audio in memory**: numpy buffers straight to Whisper; never write audio to disk except via `save_wav()`.
3. **Offline**: everything must work without internet after the first Whisper model download.
4. **Never store text persistently** (notes, history, clipboard content) unless behind an explicit opt-in setting.
5. **User feedback goes through events**; the CLI renders them with emoji prefixes (🎤 recording, 📝 transcribing, 🧠 refining, ✨ output, 📋 clipboard, ⚠️ notice, ❌ error).
6. **Never block the menu thread or the hotkey thread** with AI work; queue a job.
7. **Do not weaken error handling**: an error must never end up on the clipboard.
8. **macOS permissions**: the global hotkey needs the terminal (or the Python binary) under
   **System Settings → Privacy & Security → Accessibility** *and* **Input Monitoring**; the mic needs
   **Microphone**. Keep `README.md` in sync when touching this.
9. **Update docs in the same change**: this file for architecture and patterns, `README.md` for
   user-visible behaviour, `docs/CODE_REVIEW.md` when closing a finding.

## Common Tasks

| Task | Where |
|------|-------|
| Add a CLI menu command | `cli.py`: `print_menu()` and `show_menu()`; add an `Engine` method and an event in `events.py` if the engine must do something new |
| Add a menu bar item or change what the menu shows | `gui/presenter.py`: add a fact to `MenuModel` or a derived property, extend `_apply()`, add the command method; `gui/menubar.py`: create the `rumps.MenuItem` and apply it in `apply()`. Test the mapping in `tests/test_gui.py` |
| Make a setting editable from the menu bar | It already is if it is in `SETTING_LABELS`/`SETTING_HINTS` (`gui/menubar.py`); add it there. If it must apply live, make sure `AIProcessor` reads it per call; otherwise add it to `RESTART_SETTINGS` in `gui/presenter.py` |
| Add a new frontend | Subscribe to `Engine` events and call its command methods; `LocalWhisperCLI` and `MenuBarPresenter` are the reference presenters |
| Change a default (models, language, hotkey, URL, timeouts) | `config.py`: `Settings` field defaults; users override via file, env or flags |
| Change the refinement prompt or sampling options | `prompts.py`; keep the delimiters and the "never answer the transcript" rule |
| Change the Ollama model choice | Update `docs/OLLAMA_MODEL_DECISION.md` first, then the `Settings` default and the `ollama pull` line in `README.md` |
| Adjust audio constants | Top of `audio.py` |
| Add a dependency | `uv add ...`; never edit `uv.lock` by hand |

## Testing Checklist

Automated (CI): `ruff check`, `ruff format --check`, `mypy src`, `pytest`.

Manual, before a release or after touching `audio.py`, `cli.py` or the hotkey:
- [ ] `uv run localwhisper` starts, reports the Whisper model and "Ollama model ready"
- [ ] Hotkey starts and stops a recording from another app
- [ ] A silent recording says "No speech detected" and keeps the previous note
- [ ] Refined text lands on the clipboard; with Ollama stopped, the raw transcript does and the menu says so
- [ ] `[v]`, `[m]`, `[u]`, `[s]` behave as documented in `README.md`
- [ ] `q` and Ctrl+C exit promptly with no traceback or warning
- [ ] `uv run localwhisper-gui`: the status item shows ⏳ then 🎙; hotkey from another app flips it
      to 🔴, then 📝/🧠, a notification appears and the text lands on the clipboard; Modify with
      Voice / Modify with Text… / Undo round-trip; Settings ▸ Hotkey rebinds live and the config file
      keeps its comments; Auto-paste pastes into the frontmost app; Quit exits without a traceback.
      Ctrl+C in the terminal also exits cleanly but may print a harmless `resource_tracker: ...
      leaked semaphore` warning (tqdm's multiprocessing lock from the Hugging Face model load
      meets PyObjC's Mach SIGINT handler); the menu's Quit does not.
