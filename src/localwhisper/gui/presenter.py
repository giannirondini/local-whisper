"""Toolkit-free presenter for the menu bar app.

The presenter turns engine events into a small immutable `MenuModel` and menu
clicks into engine calls. It knows nothing about `rumps` or AppKit, which keeps
the event → menu mapping unit-testable with a fake engine. Side effects that need
a toolkit or the OS (clipboard, Cmd+V, notifications, saving the config file,
rebinding the hotkey) are injected as plain callables.

Threading contract (see `engine.py`): engine events arrive on the worker or hotkey
thread, but AppKit may only be touched from the main thread. `post()` is the
thread-safe half: it just queues. `drain()` is the main-thread half: the shell
calls it from a UI timer, applies everything queued since the last call, and
returns the model when it differs from the one last returned. Command methods run
on the main thread (menu clicks) and may update the model directly; the next
`drain()` picks that up too. The same queue carries the outcome of the background
model load, so the engine is attached on the main thread.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pyperclip

from ..config import ConfigError, Settings, coerce_setting, save_setting
from ..engine import NO_TEXT_MESSAGE, Engine
from ..events import (
    Error,
    Event,
    Notice,
    RecordingKind,
    RecordingStarted,
    RecordingStopped,
    RefinementReady,
    Refining,
    Reverted,
    State,
    StateChanged,
    Transcribing,
    TranscriptReady,
)

logger = logging.getLogger(__name__)

LOADING_ICON = "⏳"
FAILED_ICON = "⚠️"
TRANSCRIBING_ICON = "📝"
REFINING_ICON = "🧠"
ICONS: dict[State, str] = {
    State.IDLE: "🎙",
    State.RECORDING_NOTE: "🔴",
    State.RECORDING_INSTRUCTION: "🟠",
    State.PROCESSING: "⏳",
}

RECORD_TITLE = "Record"
STOP_TITLE = "Stop Recording"
INSTRUCTION_TITLE = "Modify with Voice"
STOP_INSTRUCTION_TITLE = "Stop Instruction"
LOADING_STATUS = "Loading Whisper model…"
DOWNLOAD_ICON = "⬇️"
DOWNLOAD_TITLE = "Downloading Whisper model"
DOWNLOADED_TITLE = "Whisper model downloaded"
COPIED_TITLE = "Copied to clipboard"
FALLBACK_TITLE = "Raw transcript copied"
ERROR_TITLE = "LocalWhisper error"
NOTICE_TITLE = "LocalWhisper"
PREVIEW_CHARS = 60
NOTIFICATION_CHARS = 120

RESTART_SETTINGS = frozenset({"whisper_model"})
"""Settings that are saved but only take effect on the next launch."""


@dataclass(frozen=True)
class MenuModel:
    """Facts the shell renders. Derived titles/enabled flags are properties.

    A new instance per change, so `==` detects no-ops.
    """

    icon: str
    """Text shown in the status bar."""

    status: str
    """One-line, non-clickable menu entry: the last thing worth telling the user."""

    settings: Settings

    state: State | None = None
    """Engine state; None while loading or after a failed load."""

    has_text: bool = False
    """A note exists, so the modify/undo/show/copy actions make sense."""

    @property
    def loaded(self) -> bool:
        return self.state is not None

    @property
    def record_title(self) -> str:
        return STOP_TITLE if self.state is State.RECORDING_NOTE else RECORD_TITLE

    @property
    def record_enabled(self) -> bool:
        # An instruction recording is stopped by the item that started it.
        return self.loaded and self.state is not State.RECORDING_INSTRUCTION

    @property
    def instruction_title(self) -> str:
        if self.state is State.RECORDING_INSTRUCTION:
            return STOP_INSTRUCTION_TITLE
        return INSTRUCTION_TITLE

    @property
    def instruction_enabled(self) -> bool:
        if self.state is State.RECORDING_INSTRUCTION:
            return True
        return self.state in (State.IDLE, State.PROCESSING) and self.has_text

    @property
    def text_actions_enabled(self) -> bool:
        """Modify with Text, Undo, Show Last Text, Copy Again."""
        return self.loaded and self.has_text


@dataclass(frozen=True)
class _Attached:
    engine: Engine
    warning: str | None


@dataclass(frozen=True)
class _LoadFailed:
    message: str


@dataclass(frozen=True)
class _Downloading:
    model: str
    done: int
    total: int | None


@dataclass(frozen=True)
class _Loading:
    """The model is on disk (downloaded or cached) and is being loaded into memory."""


_Message = Event | _Attached | _LoadFailed | _Downloading | _Loading


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Collapse whitespace and truncate for a one-line menu entry."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def download_status(model: str, done: int, total: int | None) -> str:
    """Status line for the first-run model download, e.g. `… 45% of 145 MB`."""
    mb = 1024 * 1024
    if total:
        percent = min(100, done * 100 // total)
        return f"Downloading Whisper model {model}… {percent}% of {total / mb:.0f} MB"
    return f"Downloading Whisper model {model}… {done / mb:.0f} MB"


def _nothing(*_args: object) -> None:
    pass


class MenuBarPresenter:
    """Event → `MenuModel` mapping and menu → engine commands. See module docstring."""

    def __init__(
        self,
        settings: Settings,
        *,
        copy: Callable[[str], None] = pyperclip.copy,
        paste: Callable[[], None] = _nothing,
        notify: Callable[[str, str], None] = _nothing,
        alert: Callable[[str, str], None] = _nothing,
        save: Callable[[str, Any], object] = save_setting,
        rebind_hotkey: Callable[[str], None] = _nothing,
    ) -> None:
        """
        Args:
            settings: The settings in effect at launch; edited through `update_setting()`.
            copy: Clipboard writer.
            paste: Presses Cmd+V in the frontmost app (only used when `auto_paste` is on).
            notify: Posts a user notification `(title, message)`.
            alert: Shows a blocking dialog `(title, message)` for failures, since a
                notification can be silently dropped and the status line is easy to miss.
            save: Persists one setting `(name, value)`; raises `ConfigError` on failure.
            rebind_hotkey: Re-registers the global hotkey; raises `ValueError` if invalid.
        """
        self._copy = copy
        self._paste = paste
        self._notify = notify
        self._alert = alert
        self._save = save
        self._rebind_hotkey = rebind_hotkey
        self._queue: queue.SimpleQueue[_Message] = queue.SimpleQueue()
        self._engine: Engine | None = None
        self.model = MenuModel(icon=LOADING_ICON, status=LOADING_STATUS, settings=settings)
        self._shown: MenuModel | None = None
        self._download_announced = False

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @property
    def settings(self) -> Settings:
        return self.model.settings

    # ------------------------------------------------------- any thread → queue

    def post(self, event: Event) -> None:
        """Engine callback. Only queues, so it is safe inside the engine lock."""
        self._queue.put(event)

    def attach(self, engine: Engine, warning: str | None = None) -> None:
        """Hand over the loaded engine. Subscribes immediately so no event is lost."""
        engine.subscribe(self.post)
        self._queue.put(_Attached(engine, warning))

    def downloading(self, model: str, done: int, total: int | None) -> None:
        """Loader thread: first-run model download progress (`done`/`total` bytes)."""
        self._queue.put(_Downloading(model, done, total))

    def loading(self) -> None:
        """Loader thread: the model files are local and are now being loaded."""
        self._queue.put(_Loading())

    def fail(self, message: str) -> None:
        """Report that the engine could not be built; the app stays up so Quit works."""
        self._queue.put(_LoadFailed(message))

    # ------------------------------------------------------------ main thread

    def drain(self) -> MenuModel | None:
        """Apply every queued message; return the model if it differs from the last returned."""
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            self.model = self._apply(self.model, message)
        if self.model == self._shown:
            return None
        self._shown = self.model
        return self.model

    # ------------------------------------------------- commands (main thread)

    def toggle_record(self) -> None:
        """Record menu item and global hotkey. No-op until the engine is attached."""
        engine = self._engine
        if engine is None:
            logger.info("Record requested before the engine was ready; ignored.")
            return
        engine.toggle()

    def toggle_instruction(self) -> None:
        """Modify with Voice: start a spoken instruction, or stop the one being recorded."""
        engine = self._engine
        if engine is None:
            return
        if engine.state is State.RECORDING_INSTRUCTION:
            engine.stop()
        elif engine.last_raw_text is None:
            self._set_status(f"⚠️ {NO_TEXT_MESSAGE}")
        else:
            engine.start(RecordingKind.INSTRUCTION)

    def modify(self, instruction: str) -> None:
        """Modify with Text: queue a typed instruction (the engine reports if there is no text)."""
        if self._engine is not None:
            self._engine.modify(instruction)

    def undo(self) -> None:
        if self._engine is not None:
            self._engine.undo()

    def last_text(self) -> tuple[str, list[str]] | None:
        """The current text and the instructions applied to it, for the Show window."""
        engine = self._engine
        if engine is None or (text := engine.last_refined_text) is None:
            return None
        return text, [r.instruction for r in engine.history if r.instruction]

    def copy_again(self) -> None:
        if (found := self.last_text()) is not None:
            self._set_status(self._deliver(found[0], "Copied again", notification=None))

    def update_setting(self, name: str, value: Any) -> None:
        """Validate, apply for this session, then persist one setting; report in the status line."""
        try:
            typed = coerce_setting(name, value)
            settings = replace(self.model.settings, **{name: typed})
        except ConfigError as e:
            self._set_status(f"❌ {e}")
            return
        if name == "hotkey":
            try:
                self._rebind_hotkey(settings.hotkey)
            except ValueError as e:
                self._set_status(f"❌ Invalid hotkey {settings.hotkey!r}: {e}")
                return

        self.model = replace(self.model, settings=settings)
        if self._engine is not None:
            self._engine.update_settings(settings)

        shown = "auto" if typed is None else typed
        try:
            self._save(name, typed)
        except ConfigError as e:
            self._set_status(f"⚠️ {name} = {shown} applied for this session only: {e}")
            return
        if name in RESTART_SETTINGS:
            self._set_status(f"💾 {name} = {shown} saved; restart to apply")
        else:
            self._set_status(f"💾 {name} = {shown}")

    def close(self, timeout: float = 5.0) -> bool:
        """Shut the engine down. True if the worker exited in time (or never existed)."""
        engine = self._engine
        if engine is None:
            return True
        return engine.shutdown(timeout)

    # -------------------------------------------------------------- internals

    def _set_status(self, status: str) -> None:
        self.model = replace(self.model, status=status)

    def _apply(self, model: MenuModel, message: _Message) -> MenuModel:
        match message:
            case _Attached(engine=engine, warning=warning):
                self._engine = engine
                hotkey = model.settings.hotkey
                status = f"⚠️ {warning}" if warning else f"Ready · hotkey {hotkey}"
                return replace(model, icon=ICONS[State.IDLE], state=State.IDLE, status=status)
            case _Downloading(model=name, done=done, total=total):
                if not self._download_announced:
                    # The one moment the app uses the network: say so, and say it is one-time.
                    self._download_announced = True
                    size = f" ({total / (1024 * 1024):.0f} MB)" if total else ""
                    body = f"One-time download of {name}{size} from Hugging Face."
                    self._notify(DOWNLOAD_TITLE, body)
                status = download_status(name, done, total)
                return replace(model, icon=DOWNLOAD_ICON, state=None, status=status)
            case _Loading():
                if self._download_announced:
                    self._notify(DOWNLOADED_TITLE, "Loading it; works offline from now on.")
                return replace(model, icon=LOADING_ICON, state=None, status=LOADING_STATUS)
            case _LoadFailed(message=text):
                # Nothing works without the model, and the app has no window: be loud.
                self._alert(ERROR_TITLE, text)
                return replace(model, icon=FAILED_ICON, state=None, status=f"❌ {text}")
            case StateChanged(current=state):
                return replace(model, icon=ICONS[state], state=state)
            case RecordingStarted(kind=RecordingKind.NOTE):
                hotkey = model.settings.hotkey
                return replace(model, status=f"Recording… {hotkey} or Stop to finish")
            case RecordingStarted(kind=RecordingKind.INSTRUCTION):
                return replace(model, status="Recording instruction… Stop Instruction to finish")
            case RecordingStopped(dropped_chunks=dropped) if dropped:
                return replace(model, status=f"⚠️ Input overflow: {dropped} chunk(s) dropped")
            case Transcribing(kind=RecordingKind.NOTE):
                return replace(model, icon=TRANSCRIBING_ICON, status="Transcribing…")
            case Transcribing(kind=RecordingKind.INSTRUCTION):
                return replace(model, icon=TRANSCRIBING_ICON, status="Transcribing instruction…")
            case TranscriptReady(kind=RecordingKind.INSTRUCTION, text=text):
                return replace(model, status=f"Instruction: {preview(text)}")
            case Refining(instruction=None):
                return replace(model, icon=REFINING_ICON, status="Refining…")
            case Refining(instruction=str(instruction)):
                status = f"Refining: {preview(instruction)}"
                return replace(model, icon=REFINING_ICON, status=status)
            case RefinementReady(text=text, fallback_raw=True):
                status = self._deliver(text, "Raw transcript copied", notification=FALLBACK_TITLE)
                return replace(model, has_text=True, status=status)
            case RefinementReady(text=text):
                status = self._deliver(text, "Copied", notification=COPIED_TITLE)
                return replace(model, has_text=True, status=status)
            case Reverted(text=text, revisions_left=left):
                label = "raw transcript" if left == 0 else f"revision {left}"
                status = self._deliver(text, f"Reverted to {label}", notification=COPIED_TITLE)
                return replace(model, has_text=True, status=status)
            case Notice(message=text):
                self._notify(NOTICE_TITLE, text)
                return replace(model, status=f"⚠️ {text}")
            case Error(message=text):
                # A notification can be silently dropped and the status line is only
                # visible with the menu open; an Error means nothing was produced, so
                # it also gets a blocking alert (F-02: never let a failure go unnoticed).
                self._notify(ERROR_TITLE, text)
                self._alert(ERROR_TITLE, text)
                return replace(model, status=f"❌ {text}")
            case _:
                return model

    def _deliver(self, text: str, label: str, notification: str | None) -> str:
        """Copy (and paste, notify) a result; returns the status line. Never raises."""
        try:
            self._copy(text)
        except pyperclip.PyperclipException as e:
            logger.warning("Clipboard copy failed: %s", e)
            self._notify(ERROR_TITLE, f"Could not copy to clipboard: {e}")
            return f"❌ Could not copy to clipboard: {e}"
        if notification is not None:
            self._notify(notification, preview(text, NOTIFICATION_CHARS))
        if self.model.settings.auto_paste:
            try:
                self._paste()
            except Exception:  # a toolkit failure must not lose the result
                logger.exception("Auto-paste failed")
                return f"📋 {label} (auto-paste failed): {preview(text)}"
        return f"📋 {label}: {preview(text)}"
