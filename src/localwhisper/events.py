"""State and event vocabulary shared by the engine and its presenters.

The engine (`engine.py`) owns a single `State` and reports everything it does as
immutable `Event` objects delivered to subscribed callbacks. Presenters (the CLI
today, a menu bar app later) turn events into output; the engine itself never
prints, copies to the clipboard, or touches a UI toolkit.

Events are delivered synchronously on whichever thread produced them: the worker
thread for processing results, the caller's thread for `start()`/`stop()`
transitions. Presenters must marshal to their own UI thread if they need one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(Enum):
    """Capture state of the engine.

    PROCESSING means "not recording, but the worker is still busy". A new recording
    may start from PROCESSING; its job simply queues behind the running one.
    """

    IDLE = "idle"
    RECORDING_NOTE = "recording_note"
    RECORDING_INSTRUCTION = "recording_instruction"
    PROCESSING = "processing"

    @property
    def is_recording(self) -> bool:
        return self in (State.RECORDING_NOTE, State.RECORDING_INSTRUCTION)


class RecordingKind(Enum):
    """What a recording is for: a new note, or a spoken instruction about the last one."""

    NOTE = "note"
    INSTRUCTION = "instruction"

    @property
    def recording_state(self) -> State:
        return State.RECORDING_NOTE if self is RecordingKind.NOTE else State.RECORDING_INSTRUCTION


@dataclass(frozen=True)
class Event:
    """Base class for everything the engine reports."""


@dataclass(frozen=True)
class StateChanged(Event):
    previous: State
    current: State


@dataclass(frozen=True)
class RecordingStarted(Event):
    kind: RecordingKind


@dataclass(frozen=True)
class RecordingStopped(Event):
    kind: RecordingKind
    seconds: float
    dropped_chunks: int


@dataclass(frozen=True)
class Transcribing(Event):
    kind: RecordingKind


@dataclass(frozen=True)
class TranscriptReady(Event):
    kind: RecordingKind
    text: str
    seconds: float
    """Wall-clock seconds Whisper took."""


@dataclass(frozen=True)
class Refining(Event):
    instruction: str | None


@dataclass(frozen=True)
class RefinementReady(Event):
    """The text a presenter should now hand to the user (clipboard, window, ...)."""

    text: str
    instruction: str | None
    fallback_raw: bool
    """True when refinement failed and `text` is the raw transcript instead."""


@dataclass(frozen=True)
class Reverted(Event):
    """An edit was undone; `text` is what the user should now have."""

    text: str
    revisions_left: int


@dataclass(frozen=True)
class Notice(Event):
    """Something worth telling the user that is not an error (no speech, hotkey ignored)."""

    message: str


@dataclass(frozen=True)
class Error(Event):
    message: str
