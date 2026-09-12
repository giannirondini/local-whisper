"""Headless orchestration engine for LocalWhisper.

The engine owns the recorder, the AI processor, one worker thread and one `State`.
Every state transition goes through `_transition()` under a single lock, and every
piece of processing (transcribe, refine, apply an instruction) runs as a job on the
worker, one at a time. Presenters call `start()`, `stop()`, `toggle()`,
`modify()` and `shutdown()`, and receive `events.Event` objects in return.

Rules that presenters can rely on:

- `start()`/`stop()`/`toggle()`/`modify()` are safe to call from any thread and
  never block on AI work.
- Jobs run strictly in FIFO order on one thread, so a spoken or typed instruction
  queued behind a note is applied to *that* note once it exists.
- `last_raw_text`/`last_refined_text` are only replaced once a non-empty result
  exists; an empty or failed recording never wipes the previous note.
- Each note keeps a revision history: the automatic clean-up is revision 1, every
  instruction (spoken or typed) is applied to the *current* text and appended, and
  `undo()` pops one revision. Undoing everything gives the raw transcript back.
- Events are delivered on the thread that produced them, inside the engine lock.
  Callbacks must be quick and must not call back into the engine.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE, AudioDeviceError, AudioRecorder
from .config import Settings
from .core import AIProcessor, RefinementError, TranscriptionError
from .events import (
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

EventCallback = Callable[[Event], None]

NO_SPEECH_MESSAGE = "No speech detected. Previous text kept."
NO_INSTRUCTION_MESSAGE = "No instruction detected. Previous text kept."
NO_AUDIO_MESSAGE = "No audio recorded."
NO_TEXT_MESSAGE = "No text to modify yet. Record something first."
NOTHING_TO_UNDO_MESSAGE = "Nothing to undo."
INSTRUCTION_WITHOUT_TEXT_MESSAGE = "No previous text to modify. Recording treated as a new note."
HOTKEY_IGNORED_MESSAGE = "Hotkey ignored while recording an instruction. Press ENTER to stop."


@dataclass(frozen=True)
class _NoteJob:
    audio: np.ndarray


@dataclass(frozen=True)
class _InstructionJob:
    audio: np.ndarray


@dataclass(frozen=True)
class _TextJob:
    instruction: str


_Job = _NoteJob | _InstructionJob | _TextJob


@dataclass(frozen=True)
class Revision:
    """One version of the current note's text and the instruction that produced it."""

    instruction: str | None
    """None for the automatic clean-up of the raw transcript."""

    text: str


class Engine:
    """Single-threaded state machine plus one worker thread. See module docstring."""

    def __init__(
        self,
        recorder: AudioRecorder,
        ai: AIProcessor,
        on_event: EventCallback | None = None,
    ) -> None:
        self._recorder = recorder
        self._ai = ai
        self._callbacks: list[EventCallback] = [on_event] if on_event else []

        self._lock = threading.RLock()
        self._state = State.IDLE
        self._closed = False
        self._raw_text: str | None = None
        self._revisions: list[Revision] = []

        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._run_worker, name="localwhisper-worker", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ queries

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def busy(self) -> bool:
        """True while the worker has a running or pending job."""
        return not self._idle.is_set()

    @property
    def last_raw_text(self) -> str | None:
        """The raw Whisper transcript of the current note."""
        with self._lock:
            return self._raw_text

    @property
    def last_refined_text(self) -> str | None:
        """The current text: the latest revision, or the raw transcript if none."""
        with self._lock:
            if self._revisions:
                return self._revisions[-1].text
            return self._raw_text

    @property
    def history(self) -> tuple[Revision, ...]:
        """Revisions of the current note, oldest first."""
        with self._lock:
            return tuple(self._revisions)

    def subscribe(self, callback: EventCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no job is running or pending. Mainly for tests and shutdown."""
        return self._idle.wait(timeout)

    # ---------------------------------------------------------------- commands

    def start(self, kind: RecordingKind = RecordingKind.NOTE) -> bool:
        """Start recording. Returns False if already recording or shut down."""
        with self._lock:
            if self._closed or self._state.is_recording:
                return False
            try:
                self._recorder.start_recording()
            except AudioDeviceError as e:
                self._emit(Error(f"Could not start recording: {e}"))
                return False
            self._transition(kind.recording_state)
            self._emit(RecordingStarted(kind))
            return True

    def stop(self) -> bool:
        """Stop recording and queue the audio for processing. False if not recording."""
        with self._lock:
            if not self._state.is_recording:
                return False
            kind = (
                RecordingKind.NOTE
                if self._state is State.RECORDING_NOTE
                else RecordingKind.INSTRUCTION
            )
            audio = self._recorder.stop_recording()
            seconds = 0.0 if audio is None else len(audio) / SAMPLE_RATE
            self._emit(RecordingStopped(kind, seconds, self._recorder.dropped_chunks))

            if audio is None:
                self._emit(Notice(NO_AUDIO_MESSAGE))
                self._transition(State.PROCESSING if self.busy else State.IDLE)
                return True

            job: _Job = _NoteJob(audio) if kind is RecordingKind.NOTE else _InstructionJob(audio)
            self._enqueue(job)
            self._transition(State.PROCESSING)
            return True

    def toggle(self) -> bool:
        """Hotkey behaviour: start a note, or stop the note being recorded.

        A recording started by the menu (an instruction) is not the hotkey's to stop.
        """
        with self._lock:
            if self._state is State.RECORDING_INSTRUCTION:
                self._emit(Notice(HOTKEY_IGNORED_MESSAGE))
                return False
            if self._state is State.RECORDING_NOTE:
                return self.stop()
            return self.start(RecordingKind.NOTE)

    def modify(self, instruction: str) -> bool:
        """Queue a typed instruction to be applied to the last note. Never blocks."""
        instruction = instruction.strip()
        if not instruction:
            return False
        with self._lock:
            if self._closed:
                return False
            if self._raw_text is None and not self.busy:
                # Nothing to modify and nothing in flight that could produce it.
                self._emit(Notice(NO_TEXT_MESSAGE))
                return False
            self._enqueue(_TextJob(instruction))
            if self._state is State.IDLE:
                self._transition(State.PROCESSING)
            return True

    def undo(self) -> bool:
        """Drop the latest revision. Returns False if there is nothing to undo."""
        with self._lock:
            if not self._revisions:
                self._emit(Notice(NOTHING_TO_UNDO_MESSAGE))
                return False
            self._revisions.pop()
            current = self.last_refined_text
            assert current is not None
            self._emit(Reverted(current, len(self._revisions)))
            return True

    def update_settings(self, settings: Settings) -> None:
        """Use `settings` for every job started from now on.

        Ollama model/URL/timeouts and the Whisper language take effect immediately; a
        different `whisper_model` does not (the loaded model is kept) and needs a
        restart. Presenters must go through here rather than touching the processor.
        """
        with self._lock:
            self._ai.settings = settings

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop recording, drop pending jobs, wait for the running one, release audio.

        Returns True if the worker exited within `timeout`. A worker stuck in a long
        inference is abandoned (it is a daemon thread) rather than blocking exit.
        """
        with self._lock:
            if self._closed:
                return not self._worker.is_alive()
            self._closed = True
            self._stopping.set()
            if self._state.is_recording:
                self._recorder.stop_recording()  # discard: the user chose to quit
            self._jobs.put(None)  # wake the worker if it is waiting
            if self._state is not State.IDLE:
                self._transition(State.IDLE)

        self._worker.join(timeout)
        alive = self._worker.is_alive()
        if alive:
            logger.warning("Worker still busy after %.1fs; abandoning it.", timeout)
        self._recorder.terminate()
        return not alive

    # ---------------------------------------------------------------- internals

    def _transition(self, new_state: State) -> None:
        previous = self._state
        if previous is new_state:
            return
        self._state = new_state
        self._emit(StateChanged(previous, new_state))

    def _emit(self, event: Event) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
            for callback in callbacks:
                try:
                    callback(event)
                except Exception:
                    logger.exception("Event callback failed for %r", event)

    def _enqueue(self, job: _Job) -> None:
        self._idle.clear()
        self._jobs.put(job)

    def _run_worker(self) -> None:
        try:
            while True:
                job = self._jobs.get()
                if job is None or self._stopping.is_set():
                    break
                try:
                    self._run_job(job)
                except Exception as e:
                    logger.exception("Unexpected error while processing %s", type(job).__name__)
                    self._emit(Error(f"Unexpected error: {e}"))
                finally:
                    self._after_job()
        finally:
            self._idle.set()

    def _after_job(self) -> None:
        with self._lock:
            if not self._jobs.empty():
                return
            self._idle.set()
            if self._state is State.PROCESSING:
                self._transition(State.IDLE)

    def _run_job(self, job: _Job) -> None:
        if isinstance(job, _NoteJob):
            self._process_note(job.audio)
        elif isinstance(job, _InstructionJob):
            self._process_instruction(job.audio)
        else:
            self._process_text(job.instruction)

    def _transcribe(self, audio: np.ndarray, kind: RecordingKind) -> str | None:
        """Run Whisper and report; None means the caller should stop (error or silence)."""
        self._emit(Transcribing(kind))
        started = time.perf_counter()
        try:
            text = self._ai.transcribe(audio)
        except TranscriptionError as e:
            self._emit(Error(f"Transcription failed: {e}"))
            return None
        if not text:
            self._emit(
                Notice(NO_SPEECH_MESSAGE if kind is RecordingKind.NOTE else NO_INSTRUCTION_MESSAGE)
            )
            return None
        self._emit(TranscriptReady(kind, text, time.perf_counter() - started))
        return text

    def _process_note(self, audio: np.ndarray) -> None:
        raw_text = self._transcribe(audio, RecordingKind.NOTE)
        if raw_text is None:
            return
        with self._lock:
            self._raw_text = raw_text
            self._revisions = []

        self._emit(Refining(None))
        try:
            refined_text = self._ai.refine_text(raw_text)
        except RefinementError as e:
            self._emit(Error(f"Refinement failed: {e}"))
            # No revision: the current text is the raw transcript.
            self._emit(RefinementReady(raw_text, instruction=None, fallback_raw=True))
            return

        with self._lock:
            self._revisions.append(Revision(None, refined_text))
        self._emit(RefinementReady(refined_text, instruction=None, fallback_raw=False))

    def _process_instruction(self, audio: np.ndarray) -> None:
        if self._raw_text is None:
            self._emit(Notice(INSTRUCTION_WITHOUT_TEXT_MESSAGE))
            self._process_note(audio)
            return
        instruction = self._transcribe(audio, RecordingKind.INSTRUCTION)
        if instruction is None:
            return
        self._apply_instruction(instruction)

    def _process_text(self, instruction: str) -> None:
        if self._raw_text is None:
            self._emit(Notice(NO_TEXT_MESSAGE))
            return
        self._apply_instruction(instruction)

    def _apply_instruction(self, instruction: str) -> None:
        """Apply an instruction to the current text (F-15: edits compose)."""
        base = self.last_refined_text
        assert base is not None
        self._emit(Refining(instruction))
        try:
            new_text = self._ai.refine_text(base, instruction=instruction)
        except RefinementError as e:
            self._emit(Error(f"Refinement failed: {e}. Previous text kept."))
            return
        with self._lock:
            self._revisions.append(Revision(instruction, new_text))
        self._emit(RefinementReady(new_text, instruction=instruction, fallback_raw=False))
