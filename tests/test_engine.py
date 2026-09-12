"""Tests for the headless engine (review session S3).

The recorder and the AI processor are replaced by small fakes; the worker thread is
real, so these tests also exercise the locking and the FIFO job queue.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from localwhisper.audio import SAMPLE_RATE, AudioDeviceError
from localwhisper.core import RefinementError, TranscriptionError
from localwhisper.engine import (
    HOTKEY_IGNORED_MESSAGE,
    NO_AUDIO_MESSAGE,
    NO_SPEECH_MESSAGE,
    NO_TEXT_MESSAGE,
    NOTHING_TO_UNDO_MESSAGE,
    Engine,
    Revision,
)
from localwhisper.events import (
    Error,
    Notice,
    RecordingKind,
    RecordingStarted,
    RecordingStopped,
    RefinementReady,
    Reverted,
    State,
    StateChanged,
    TranscriptReady,
)

AUDIO = np.zeros(SAMPLE_RATE, dtype=np.float32)  # one second
TIMEOUT = 5.0


class FakeRecorder:
    """Enforces the start/stop protocol: a double start or a stray stop is a bug."""

    def __init__(self) -> None:
        self.recording = False
        self.dropped_chunks = 0
        self.audio: np.ndarray | None = AUDIO
        self.starts = 0
        self.stops = 0
        self.terminated = False
        self.fail_start: Exception | None = None

    def start_recording(self) -> None:
        if self.fail_start is not None:
            raise self.fail_start
        assert not self.recording, "start_recording() called while already recording"
        self.recording = True
        self.starts += 1

    def stop_recording(self) -> np.ndarray | None:
        assert self.recording, "stop_recording() called while not recording"
        self.recording = False
        self.stops += 1
        return self.audio

    def terminate(self) -> None:
        self.terminated = True


class FakeAI:
    """Records concurrency and the calling thread so serialisation can be asserted."""

    def __init__(self) -> None:
        self.transcribe = MagicMock(side_effect=self._transcribe)
        self.refine_text = MagicMock(side_effect=self._refine)
        self.transcript = "hello world"
        self.refined = "Hello, world."
        self.transcribe_error: Exception | None = None
        self.refine_error: Exception | None = None
        self.gate: threading.Event | None = None  # transcribe blocks on this if set
        self.started = threading.Event()  # set once any AI call has begun
        self.in_flight = 0
        self.max_in_flight = 0
        self.threads: set[str] = set()
        self._lock = threading.Lock()

    def _enter(self) -> None:
        self.started.set()
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.threads.add(threading.current_thread().name)

    def _exit(self) -> None:
        with self._lock:
            self.in_flight -= 1

    def _transcribe(self, audio: np.ndarray) -> str:
        self._enter()
        try:
            if self.gate is not None:
                self.gate.wait(TIMEOUT)
            if self.transcribe_error is not None:
                raise self.transcribe_error
            return self.transcript
        finally:
            self._exit()

    def _refine(self, raw_text: str, instruction: str | None = None) -> str:
        self._enter()
        try:
            if self.refine_error is not None:
                raise self.refine_error
            return self.refined if instruction is None else f"{raw_text} [{instruction}]"
        finally:
            self._exit()


@pytest.fixture
def recorder() -> FakeRecorder:
    return FakeRecorder()


@pytest.fixture
def ai() -> FakeAI:
    return FakeAI()


@pytest.fixture
def events() -> list:
    return []


@pytest.fixture
def engine(recorder, ai, events):
    eng = Engine(recorder, ai, on_event=events.append)
    yield eng
    eng.shutdown(timeout=TIMEOUT)


def record_note(engine: Engine) -> None:
    assert engine.start() is True
    assert engine.stop() is True
    assert engine.wait_idle(TIMEOUT)


def states(events: list) -> list[State]:
    return [e.current for e in events if isinstance(e, StateChanged)]


class TestStateMachine:
    def test_initial_state(self, engine):
        assert engine.state is State.IDLE
        assert engine.busy is False
        assert engine.last_raw_text is None
        assert engine.last_refined_text is None

    def test_start_records_and_emits(self, engine, recorder, events):
        assert engine.start() is True

        assert engine.state is State.RECORDING_NOTE
        assert recorder.recording is True
        assert events == [
            StateChanged(State.IDLE, State.RECORDING_NOTE),
            RecordingStarted(RecordingKind.NOTE),
        ]

    def test_start_instruction(self, engine, events):
        engine.start(RecordingKind.INSTRUCTION)

        assert engine.state is State.RECORDING_INSTRUCTION
        assert RecordingStarted(RecordingKind.INSTRUCTION) in events

    def test_double_start_is_rejected(self, engine, recorder):
        """F-06: a second start never reaches the recorder."""
        assert engine.start() is True
        assert engine.start() is False
        assert engine.start(RecordingKind.INSTRUCTION) is False
        assert recorder.starts == 1

    def test_stop_when_idle_is_rejected(self, engine, recorder):
        """F-06: a stray stop never reaches the recorder."""
        assert engine.stop() is False
        assert recorder.stops == 0

    def test_full_note_cycle(self, engine, ai, events):
        record_note(engine)

        assert engine.state is State.IDLE
        assert engine.busy is False
        assert engine.last_raw_text == "hello world"
        assert engine.last_refined_text == "Hello, world."
        assert states(events) == [
            State.RECORDING_NOTE,
            State.PROCESSING,
            State.IDLE,
        ]
        assert RecordingStopped(RecordingKind.NOTE, 1.0, 0) in events
        assert RefinementReady("Hello, world.", instruction=None, fallback_raw=False) in events
        ready = next(e for e in events if isinstance(e, TranscriptReady))
        assert ready.kind is RecordingKind.NOTE
        assert ready.text == "hello world"

    def test_toggle_starts_then_stops(self, engine, recorder):
        assert engine.toggle() is True
        assert engine.state is State.RECORDING_NOTE
        assert engine.toggle() is True
        assert engine.state is State.PROCESSING
        assert recorder.starts == 1 and recorder.stops == 1

    def test_toggle_ignored_while_recording_instruction(self, engine, recorder, events):
        """F-10: the hotkey lock is a rule of the transition table, not a flag."""
        engine.start(RecordingKind.INSTRUCTION)

        assert engine.toggle() is False

        assert engine.state is State.RECORDING_INSTRUCTION
        assert recorder.stops == 0
        assert Notice(HOTKEY_IGNORED_MESSAGE) in events

    def test_toggle_from_processing_starts_new_recording(self, engine, ai):
        ai.gate = threading.Event()
        engine.start()
        engine.stop()
        assert engine.state is State.PROCESSING

        assert engine.toggle() is True
        assert engine.state is State.RECORDING_NOTE
        assert engine.busy is True

        ai.gate.set()

    def test_stop_without_audio_notices_and_returns_idle(self, engine, recorder, events):
        recorder.audio = None
        engine.start()

        assert engine.stop() is True

        assert engine.state is State.IDLE
        assert Notice(NO_AUDIO_MESSAGE) in events
        assert engine.busy is False

    def test_recorder_failure_is_reported_and_state_stays_idle(self, engine, recorder, events):
        recorder.fail_start = AudioDeviceError("no input device")

        assert engine.start() is False

        assert engine.state is State.IDLE
        assert any(isinstance(e, Error) and "no input device" in e.message for e in events)
        assert engine.stop() is False

    def test_dropped_chunks_are_surfaced(self, engine, recorder, events):
        engine.start()
        recorder.dropped_chunks = 3
        engine.stop()

        stopped = next(e for e in events if isinstance(e, RecordingStopped))
        assert stopped.dropped_chunks == 3

    def test_concurrent_toggles_never_break_the_recorder_protocol(self, recorder, ai):
        """F-06: hammering toggle() from many threads yields balanced start/stop calls."""
        ai.transcript = ""  # keep jobs trivial
        engine = Engine(recorder, ai)
        go = threading.Event()

        def hammer() -> None:
            go.wait(TIMEOUT)
            for _ in range(20):
                engine.toggle()

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        go.set()
        for t in threads:
            t.join(TIMEOUT)

        # FakeRecorder asserts on protocol violations; also check the books balance.
        assert recorder.starts - recorder.stops in (0, 1)
        engine.shutdown(timeout=TIMEOUT)


class TestProcessing:
    def test_jobs_are_serialised_on_one_worker(self, engine, ai):
        """F-08: two recordings never run two inferences at once."""
        ai.gate = threading.Event()
        record_pair = []
        for _ in range(2):
            engine.start()
            engine.stop()
            record_pair.append(engine.state)
        assert record_pair == [State.PROCESSING, State.PROCESSING]

        ai.gate.set()
        assert engine.wait_idle(TIMEOUT)

        assert ai.max_in_flight == 1
        assert ai.transcribe.call_count == 2
        assert ai.threads == {"localwhisper-worker"}
        assert engine.state is State.IDLE

    def test_state_returns_to_idle_only_after_all_jobs(self, engine, ai, events):
        ai.gate = threading.Event()
        engine.start()
        engine.stop()
        engine.start()  # allowed while processing
        engine.stop()

        ai.gate.set()
        assert engine.wait_idle(TIMEOUT)

        assert states(events).count(State.IDLE) == 1
        assert states(events)[-1] is State.IDLE

    def test_empty_transcript_keeps_previous_note(self, engine, ai, events):
        """F-01 at the engine level."""
        record_note(engine)
        ai.transcript = ""
        ai.refine_text.reset_mock()

        record_note(engine)

        assert engine.last_raw_text == "hello world"
        assert engine.last_refined_text == "Hello, world."
        ai.refine_text.assert_not_called()
        assert Notice(NO_SPEECH_MESSAGE) in events
        assert engine.state is State.IDLE

    def test_transcription_error_keeps_state(self, engine, ai, events):
        record_note(engine)
        ai.transcribe_error = TranscriptionError("boom")

        record_note(engine)

        assert engine.last_raw_text == "hello world"
        assert any(isinstance(e, Error) and "boom" in e.message for e in events)
        assert engine.state is State.IDLE

    def test_refinement_error_falls_back_to_raw(self, engine, ai, events):
        """F-02: presenters get the raw text flagged as a fallback, never an error string."""
        ai.refine_error = RefinementError("Cannot reach Ollama")

        record_note(engine)

        assert engine.last_raw_text == "hello world"
        assert engine.last_refined_text == "hello world"
        assert RefinementReady("hello world", instruction=None, fallback_raw=True) in events
        assert any(isinstance(e, Error) and "Cannot reach Ollama" in e.message for e in events)

    def test_unexpected_exception_does_not_kill_worker(self, engine, ai, events):
        ai.transcribe_error = RuntimeError("unexpected")
        record_note(engine)
        assert any(isinstance(e, Error) and "unexpected" in e.message for e in events)
        assert engine.state is State.IDLE

        ai.transcribe_error = None
        record_note(engine)

        assert engine.last_refined_text == "Hello, world."

    def test_broken_callback_does_not_kill_worker(self, recorder, ai):
        def explode(event) -> None:
            raise ValueError("presenter bug")

        engine = Engine(recorder, ai, on_event=explode)
        try:
            record_note(engine)
            assert engine.last_refined_text == "Hello, world."
            assert engine.state is State.IDLE
        finally:
            engine.shutdown(timeout=TIMEOUT)


class TestInstructions:
    def test_voice_instruction_applies_to_raw_text(self, engine, ai, events):
        record_note(engine)
        ai.transcript = "make it a list"

        engine.start(RecordingKind.INSTRUCTION)
        engine.stop()
        assert engine.wait_idle(TIMEOUT)

        ai.refine_text.assert_called_with("Hello, world.", instruction="make it a list")
        assert engine.last_refined_text == "Hello, world. [make it a list]"
        assert engine.last_raw_text == "hello world"
        heard = [e for e in events if isinstance(e, TranscriptReady)][-1]
        assert heard.kind is RecordingKind.INSTRUCTION

    def test_empty_voice_instruction_keeps_text(self, engine, ai):
        record_note(engine)
        ai.transcript = ""
        ai.refine_text.reset_mock()

        engine.start(RecordingKind.INSTRUCTION)
        engine.stop()
        assert engine.wait_idle(TIMEOUT)

        assert engine.last_refined_text == "Hello, world."
        ai.refine_text.assert_not_called()

    def test_voice_instruction_without_text_becomes_a_note(self, engine, ai):
        engine.start(RecordingKind.INSTRUCTION)
        engine.stop()
        assert engine.wait_idle(TIMEOUT)

        ai.refine_text.assert_called_once_with("hello world")
        assert engine.last_refined_text == "Hello, world."

    def test_refinement_error_keeps_previous_refined_text(self, engine, ai, events):
        record_note(engine)
        ai.refine_error = RefinementError("down")

        assert engine.modify("shorten") is True
        assert engine.wait_idle(TIMEOUT)

        assert engine.last_refined_text == "Hello, world."
        assert any(isinstance(e, Error) and "down" in e.message for e in events)

    def test_modify_runs_on_worker_not_caller(self, engine, ai):
        """F-17: a typed instruction never blocks the caller's thread."""
        record_note(engine)
        ai.gate = threading.Event()  # gate only affects transcribe, refine is instant

        assert engine.modify("translate to Italian") is True
        assert engine.wait_idle(TIMEOUT)

        assert engine.last_refined_text == "Hello, world. [translate to Italian]"
        assert threading.current_thread().name not in ai.threads

    def test_modify_blank_is_rejected(self, engine, ai):
        assert engine.modify("   ") is False
        assert engine.busy is False

    def test_modify_without_text_notices(self, engine, ai, events):
        assert engine.modify("shorten") is False
        assert Notice(NO_TEXT_MESSAGE) in events
        assert engine.state is State.IDLE

    def test_modify_queued_behind_in_flight_note_applies_to_it(self, engine, ai):
        """A typed instruction sent while a note is processing waits for that note."""
        ai.gate = threading.Event()
        engine.start()
        engine.stop()

        assert engine.modify("shorten") is True
        ai.gate.set()
        assert engine.wait_idle(TIMEOUT)

        assert engine.last_refined_text == "Hello, world. [shorten]"


class TestHistory:
    """F-15: edits compose; undo walks back to the raw transcript."""

    def test_new_note_starts_with_one_revision(self, engine):
        record_note(engine)

        assert engine.history == (Revision(None, "Hello, world."),)
        assert engine.last_refined_text == "Hello, world."

    def test_instructions_compose(self, engine, ai):
        record_note(engine)
        engine.modify("make it a list")
        assert engine.wait_idle(TIMEOUT)
        engine.modify("translate to Italian")
        assert engine.wait_idle(TIMEOUT)

        assert [r.instruction for r in engine.history] == [
            None,
            "make it a list",
            "translate to Italian",
        ]
        # The second instruction was applied to the output of the first.
        ai.refine_text.assert_called_with(
            "Hello, world. [make it a list]", instruction="translate to Italian"
        )
        assert engine.last_refined_text == "Hello, world. [make it a list] [translate to Italian]"

    def test_failed_instruction_leaves_history_untouched(self, engine, ai):
        record_note(engine)
        ai.refine_error = RefinementError("down")

        engine.modify("shorten")
        assert engine.wait_idle(TIMEOUT)

        assert len(engine.history) == 1

    def test_refinement_fallback_has_no_revision(self, engine, ai):
        ai.refine_error = RefinementError("down")
        record_note(engine)

        assert engine.history == ()
        assert engine.last_refined_text == "hello world"  # raw is the current text

    def test_new_note_resets_history(self, engine):
        record_note(engine)
        engine.modify("shorten")
        assert engine.wait_idle(TIMEOUT)

        record_note(engine)

        assert len(engine.history) == 1

    def test_undo_pops_latest_revision_and_reports(self, engine, events):
        record_note(engine)
        engine.modify("shorten")
        assert engine.wait_idle(TIMEOUT)

        assert engine.undo() is True

        assert engine.last_refined_text == "Hello, world."
        assert Reverted("Hello, world.", revisions_left=1) in events

    def test_undo_everything_returns_raw_transcript(self, engine, events):
        record_note(engine)

        assert engine.undo() is True

        assert engine.last_refined_text == "hello world"
        assert engine.history == ()
        assert Reverted("hello world", revisions_left=0) in events

    def test_undo_with_nothing_to_undo(self, engine, events):
        assert engine.undo() is False
        assert Notice(NOTHING_TO_UNDO_MESSAGE) in events

        record_note(engine)
        engine.undo()
        assert engine.undo() is False

    def test_instruction_after_undo_applies_to_reverted_text(self, engine, ai):
        record_note(engine)
        engine.undo()  # back to raw

        engine.modify("shorten")
        assert engine.wait_idle(TIMEOUT)

        ai.refine_text.assert_called_with("hello world", instruction="shorten")


class TestShutdown:
    def test_shutdown_releases_recorder_and_stops_worker(self, recorder, ai):
        engine = Engine(recorder, ai)

        assert engine.shutdown(timeout=TIMEOUT) is True

        assert recorder.terminated is True
        assert engine._worker.is_alive() is False

    def test_shutdown_while_recording_discards_audio(self, recorder, ai, events):
        engine = Engine(recorder, ai, on_event=events.append)
        engine.start()

        engine.shutdown(timeout=TIMEOUT)

        assert recorder.recording is False
        assert engine.state is State.IDLE
        ai.transcribe.assert_not_called()

    def test_shutdown_drops_pending_jobs(self, recorder, ai):
        ai.gate = threading.Event()
        engine = Engine(recorder, ai)
        engine.start()
        engine.stop()
        assert ai.started.wait(TIMEOUT)  # first job is now running
        engine.start()
        engine.stop()  # pending

        ai.gate.set()
        assert engine.shutdown(timeout=TIMEOUT) is True

        assert ai.transcribe.call_count == 1

    def test_shutdown_times_out_on_stuck_job(self, recorder, ai):
        ai.gate = threading.Event()
        engine = Engine(recorder, ai)
        engine.start()
        engine.stop()
        assert ai.started.wait(TIMEOUT)  # the job is now stuck inside transcribe()

        assert engine.shutdown(timeout=0.05) is False
        assert recorder.terminated is True

        ai.gate.set()
        engine._worker.join(TIMEOUT)

    def test_commands_after_shutdown_are_rejected(self, recorder, ai):
        engine = Engine(recorder, ai)
        engine.shutdown(timeout=TIMEOUT)

        assert engine.start() is False
        assert engine.modify("x") is False
        assert engine.shutdown(timeout=TIMEOUT) is True  # idempotent


class TestUpdateSettings:
    def test_new_settings_reach_the_processor(self, engine, ai):
        from dataclasses import replace

        from localwhisper.config import Settings

        ai.settings = Settings()
        new = replace(Settings(), ollama_model="x:1b")
        engine.update_settings(new)
        assert ai.settings is new
