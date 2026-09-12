"""Tests for the menu bar presenter (UI plan sessions U1/U2).

The engine is a mock: these tests check that engine events become the right
`MenuModel`, clipboard writes and notifications, and that menu actions become the
right engine calls. The rumps shell is exercised only when `rumps` is installed
(the `gui` extra).
"""

from __future__ import annotations

import os
import threading
from dataclasses import replace
from unittest.mock import MagicMock, call

import pyperclip
import pytest

from localwhisper.config import ConfigError, Settings
from localwhisper.engine import NO_TEXT_MESSAGE, Engine, Revision
from localwhisper.events import (
    Error,
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
from localwhisper.gui.presenter import (
    COPIED_TITLE,
    ERROR_TITLE,
    FAILED_ICON,
    FALLBACK_TITLE,
    ICONS,
    INSTRUCTION_TITLE,
    LOADING_ICON,
    LOADING_STATUS,
    NOTICE_TITLE,
    RECORD_TITLE,
    REFINING_ICON,
    STOP_INSTRUCTION_TITLE,
    STOP_TITLE,
    TRANSCRIBING_ICON,
    MenuBarPresenter,
    MenuModel,
    preview,
)

SETTINGS = Settings(hotkey="<cmd>+<shift>+g")
HOTKEY = SETTINGS.hotkey


@pytest.fixture
def engine():
    eng = MagicMock(spec=Engine)
    eng.state = State.IDLE
    eng.last_raw_text = None
    eng.last_refined_text = None
    eng.history = ()
    return eng


@pytest.fixture
def side_effects():
    return {
        "copy": MagicMock(),
        "paste": MagicMock(),
        "notify": MagicMock(),
        "alert": MagicMock(),
        "save": MagicMock(),
        "rebind_hotkey": MagicMock(),
    }


@pytest.fixture
def presenter(side_effects):
    return MenuBarPresenter(SETTINGS, **side_effects)


@pytest.fixture
def ready(presenter, engine):
    """A presenter with the engine attached and the attach message drained."""
    presenter.attach(engine)
    presenter.drain()
    return presenter


@pytest.fixture
def with_text(ready, engine):
    """A presenter that has delivered one note."""
    engine.last_raw_text = "hello"
    engine.last_refined_text = "Hello."
    engine.history = (Revision(None, "Hello."),)
    apply(ready, RefinementReady("Hello.", instruction=None, fallback_raw=False))
    return ready


def apply(presenter, *events):
    for event in events:
        presenter.post(event)
    presenter.drain()
    return presenter.model


class TestLoading:
    def test_starts_loading_with_everything_disabled(self, presenter):
        model = presenter.model
        assert model == MenuModel(icon=LOADING_ICON, status=LOADING_STATUS, settings=SETTINGS)
        assert not model.loaded
        assert not model.record_enabled
        assert not model.instruction_enabled
        assert not model.text_actions_enabled
        assert presenter.engine is None

    def test_attach_subscribes_and_enables_record(self, presenter, engine):
        presenter.attach(engine)
        engine.subscribe.assert_called_once_with(presenter.post)
        assert presenter.engine is None, "the engine is attached on the main thread, in drain()"

        model = presenter.drain()

        assert presenter.engine is engine
        assert model is not None
        assert model.icon == ICONS[State.IDLE]
        assert model.state is State.IDLE
        assert model.record_enabled
        assert not model.text_actions_enabled, "no note yet"
        assert HOTKEY in model.status

    def test_attach_shows_ollama_warning(self, presenter, engine):
        presenter.attach(engine, warning="Cannot reach Ollama")
        assert presenter.drain().status == "⚠️ Cannot reach Ollama"

    def test_load_failure_keeps_everything_disabled(self, presenter):
        presenter.fail("Could not load Whisper model: boom")
        model = presenter.drain()
        assert model.icon == FAILED_ICON
        assert not model.loaded
        assert not model.record_enabled
        assert model.status == "❌ Could not load Whisper model: boom"

    def test_commands_before_attach_are_ignored(self, presenter, engine):
        presenter.toggle_record()
        presenter.toggle_instruction()
        presenter.modify("x")
        presenter.undo()
        presenter.copy_again()
        assert presenter.last_text() is None
        assert not engine.method_calls

    def test_close_without_engine_is_fine(self, presenter):
        assert presenter.close() is True


class TestDrain:
    def test_nothing_queued_returns_none(self, ready):
        assert ready.drain() is None

    def test_unrendered_event_returns_none(self, ready):
        ready.post(TranscriptReady(RecordingKind.NOTE, "hi", 0.1))
        assert ready.drain() is None

    def test_applies_events_in_order(self, ready):
        model = apply(
            ready,
            StateChanged(State.IDLE, State.RECORDING_NOTE),
            RecordingStarted(RecordingKind.NOTE),
            StateChanged(State.RECORDING_NOTE, State.PROCESSING),
            Transcribing(RecordingKind.NOTE),
        )
        assert model.icon == TRANSCRIBING_ICON
        assert model.record_title == RECORD_TITLE
        assert model.status == "Transcribing…"

    def test_main_thread_command_change_is_returned_by_next_drain(self, ready, engine):
        """Commands mutate the model directly; the shell learns about it on the next tick."""
        ready.toggle_instruction()  # no text → status changes without an event
        model = ready.drain()
        assert model is not None
        assert model.status == f"⚠️ {NO_TEXT_MESSAGE}"
        assert ready.drain() is None

    def test_post_from_another_thread_is_applied_on_drain(self, ready):
        thread = threading.Thread(target=ready.post, args=(Notice("from worker"),))
        thread.start()
        thread.join()
        assert ready.drain().status == "⚠️ from worker"


class TestStateChanged:
    @pytest.mark.parametrize(
        ("state", "title", "enabled"),
        [
            (State.IDLE, RECORD_TITLE, True),
            (State.RECORDING_NOTE, STOP_TITLE, True),
            (State.RECORDING_INSTRUCTION, RECORD_TITLE, False),
            (State.PROCESSING, RECORD_TITLE, True),
        ],
    )
    def test_icon_and_record_item(self, ready, state, title, enabled):
        model = apply(ready, StateChanged(State.PROCESSING, state))
        assert model.icon == ICONS[state]
        assert model.record_title == title
        assert model.record_enabled is enabled

    @pytest.mark.parametrize(
        ("state", "title", "enabled"),
        [
            (State.IDLE, INSTRUCTION_TITLE, True),
            (State.PROCESSING, INSTRUCTION_TITLE, True),
            (State.RECORDING_NOTE, INSTRUCTION_TITLE, False),
            (State.RECORDING_INSTRUCTION, STOP_INSTRUCTION_TITLE, True),
        ],
    )
    def test_instruction_item_with_text(self, with_text, state, title, enabled):
        model = apply(with_text, StateChanged(State.PROCESSING, state))
        assert model.instruction_title == title
        assert model.instruction_enabled is enabled

    def test_instruction_item_without_text(self, ready):
        model = apply(ready, StateChanged(State.PROCESSING, State.IDLE))
        assert not model.instruction_enabled
        assert not model.text_actions_enabled

    def test_state_change_keeps_status(self, with_text):
        """The result line must survive the PROCESSING → IDLE transition that follows it."""
        model = apply(with_text, StateChanged(State.PROCESSING, State.IDLE))
        assert model.status == "📋 Copied: Hello."
        assert model.icon == ICONS[State.IDLE]
        assert model.text_actions_enabled


class TestStatusLine:
    def test_recording_note_mentions_hotkey(self, ready):
        assert HOTKEY in apply(ready, RecordingStarted(RecordingKind.NOTE)).status

    def test_recording_instruction(self, ready):
        model = apply(ready, RecordingStarted(RecordingKind.INSTRUCTION))
        assert model.status.startswith("Recording instruction…")

    def test_dropped_chunks_are_reported(self, ready):
        model = apply(ready, RecordingStopped(RecordingKind.NOTE, 2.0, dropped_chunks=3))
        assert "3 chunk(s) dropped" in model.status

    def test_clean_stop_is_silent(self, ready):
        ready.post(RecordingStopped(RecordingKind.NOTE, 2.0, dropped_chunks=0))
        assert ready.drain() is None

    def test_transcribing_instruction(self, ready):
        model = apply(ready, Transcribing(RecordingKind.INSTRUCTION))
        assert model.icon == TRANSCRIBING_ICON
        assert model.status == "Transcribing instruction…"

    def test_instruction_transcript_is_shown(self, ready):
        model = apply(ready, TranscriptReady(RecordingKind.INSTRUCTION, "make it a list", 0.3))
        assert model.status == "Instruction: make it a list"

    def test_refining(self, ready):
        model = apply(ready, Refining(None))
        assert model.icon == REFINING_ICON
        assert model.status == "Refining…"

    def test_refining_with_instruction(self, ready):
        assert apply(ready, Refining("shorter")).status == "Refining: shorter"

    def test_notice_and_error_prefixes(self, ready):
        assert apply(ready, Notice("No speech detected.")).status == "⚠️ No speech detected."
        assert apply(ready, Error("Transcription failed: x")).status == "❌ Transcription failed: x"


class TestDelivery:
    def test_refinement_ready_copies_and_notifies(self, ready, side_effects):
        model = apply(ready, RefinementReady("Hello.", instruction=None, fallback_raw=False))
        side_effects["copy"].assert_called_once_with("Hello.")
        side_effects["notify"].assert_called_once_with(COPIED_TITLE, "Hello.")
        side_effects["paste"].assert_not_called()
        assert model.status == "📋 Copied: Hello."
        assert model.has_text

    def test_fallback_copies_raw_and_says_so(self, ready, side_effects):
        """F-02: the raw transcript is copied and the user is told why."""
        model = apply(
            ready,
            Error("Refinement failed: Cannot reach Ollama"),
            RefinementReady("raw words", instruction=None, fallback_raw=True),
        )
        side_effects["copy"].assert_called_once_with("raw words")
        assert side_effects["notify"].call_args_list == [
            call(ERROR_TITLE, "Refinement failed: Cannot reach Ollama"),
            call(FALLBACK_TITLE, "raw words"),
        ]
        assert model.status == "📋 Raw transcript copied: raw words"

    def test_edit_result_copies(self, ready, side_effects):
        model = apply(ready, RefinementReady("- Hello.", instruction="list", fallback_raw=False))
        side_effects["copy"].assert_called_once_with("- Hello.")
        assert model.status == "📋 Copied: - Hello."

    def test_reverted_copies(self, ready, side_effects):
        model = apply(ready, Reverted("Hello.", revisions_left=1))
        side_effects["copy"].assert_called_once_with("Hello.")
        side_effects["notify"].assert_called_once_with(COPIED_TITLE, "Hello.")
        assert model.status == "📋 Reverted to revision 1: Hello."

    def test_reverted_to_raw(self, ready):
        model = apply(ready, Reverted("hello", revisions_left=0))
        assert model.status == "📋 Reverted to raw transcript: hello"

    def test_notice_is_notified(self, ready, side_effects):
        apply(ready, Notice("No speech detected."))
        side_effects["notify"].assert_called_once_with(NOTICE_TITLE, "No speech detected.")
        side_effects["alert"].assert_not_called(), "a Notice is routine; only Error blocks"

    def test_error_shows_a_blocking_alert(self, ready, side_effects):
        """The regression this covers: an instruction refine failure emits only Error, no
        RefinementReady, so the clipboard is never rewritten (F-02). A notification can be
        silently dropped and the status line needs the menu open, so Error also gets an
        alert that cannot be missed."""
        apply(ready, Error("Refinement failed: Ollama timed out after 120s. Previous text kept."))
        side_effects["alert"].assert_called_once_with(
            ERROR_TITLE, "Refinement failed: Ollama timed out after 120s. Previous text kept."
        )

    def test_copy_failure_is_reported_not_raised(self, ready, side_effects):
        side_effects["copy"].side_effect = pyperclip.PyperclipException("no pbcopy")
        model = apply(ready, RefinementReady("Hello.", instruction=None, fallback_raw=False))
        assert model.status == "❌ Could not copy to clipboard: no pbcopy"
        side_effects["notify"].assert_called_once_with(
            ERROR_TITLE, "Could not copy to clipboard: no pbcopy"
        )
        side_effects["paste"].assert_not_called()

    def test_long_text_is_previewed(self, ready, side_effects):
        text = "word " * 50
        model = apply(ready, RefinementReady(text, instruction=None, fallback_raw=False))
        side_effects["copy"].assert_called_once_with(text)  # the clipboard gets the full text
        assert len(model.status) < 80
        assert model.status.endswith("…")
        body = side_effects["notify"].call_args.args[1]
        assert len(body) <= 120


class TestAutoPaste:
    @pytest.fixture
    def pasting(self, side_effects, engine):
        presenter = MenuBarPresenter(replace(SETTINGS, auto_paste=True), **side_effects)
        presenter.attach(engine)
        presenter.drain()
        return presenter

    def test_pastes_after_copy(self, pasting, side_effects):
        apply(pasting, RefinementReady("Hello.", instruction=None, fallback_raw=False))
        side_effects["copy"].assert_called_once_with("Hello.")
        side_effects["paste"].assert_called_once_with()

    def test_pastes_after_undo(self, pasting, side_effects):
        apply(pasting, Reverted("hello", revisions_left=0))
        side_effects["paste"].assert_called_once_with()

    def test_no_paste_when_copy_failed(self, pasting, side_effects):
        side_effects["copy"].side_effect = pyperclip.PyperclipException("x")
        apply(pasting, RefinementReady("Hello.", instruction=None, fallback_raw=False))
        side_effects["paste"].assert_not_called()

    def test_paste_failure_keeps_the_result(self, pasting, side_effects):
        side_effects["paste"].side_effect = RuntimeError("no accessibility")
        model = apply(pasting, RefinementReady("Hello.", instruction=None, fallback_raw=False))
        assert model.status == "📋 Copied (auto-paste failed): Hello."

    def test_toggle_via_update_setting(self, ready, side_effects):
        ready.update_setting("auto_paste", True)
        assert ready.settings.auto_paste is True
        apply(ready, RefinementReady("Hello.", instruction=None, fallback_raw=False))
        side_effects["paste"].assert_called_once_with()


class TestCommands:
    def test_toggle_record_calls_engine(self, ready, engine):
        ready.toggle_record()
        engine.toggle.assert_called_once_with()

    def test_instruction_starts_when_text_exists(self, with_text, engine):
        with_text.toggle_instruction()
        engine.start.assert_called_once_with(RecordingKind.INSTRUCTION)
        engine.stop.assert_not_called()

    def test_instruction_stops_when_recording_one(self, with_text, engine):
        engine.state = State.RECORDING_INSTRUCTION
        with_text.toggle_instruction()
        engine.stop.assert_called_once_with()
        engine.start.assert_not_called()

    def test_instruction_without_text_is_refused(self, ready, engine):
        ready.toggle_instruction()
        engine.start.assert_not_called()
        assert ready.model.status == f"⚠️ {NO_TEXT_MESSAGE}"

    def test_modify_queues_instruction(self, with_text, engine):
        with_text.modify("shorter")
        engine.modify.assert_called_once_with("shorter")

    def test_undo(self, with_text, engine):
        with_text.undo()
        engine.undo.assert_called_once_with()

    def test_last_text_with_history(self, with_text, engine):
        engine.history = (Revision(None, "Hello."), Revision("list", "- Hello."))
        engine.last_refined_text = "- Hello."
        assert with_text.last_text() == ("- Hello.", ["list"])

    def test_last_text_without_note(self, ready):
        assert ready.last_text() is None

    def test_copy_again_copies_without_notification(self, with_text, side_effects):
        side_effects["copy"].reset_mock()
        side_effects["notify"].reset_mock()
        with_text.copy_again()
        side_effects["copy"].assert_called_once_with("Hello.")
        side_effects["notify"].assert_not_called()
        assert with_text.model.status == "📋 Copied again: Hello."

    def test_close_shuts_engine_down(self, ready, engine):
        engine.shutdown.return_value = False
        assert ready.close(timeout=1.0) is False
        engine.shutdown.assert_called_once_with(1.0)


class TestSettings:
    def test_live_setting_is_applied_and_saved(self, ready, engine, side_effects):
        ready.update_setting("ollama_model", "qwen3.5:4b-mlx")

        assert ready.settings.ollama_model == "qwen3.5:4b-mlx"
        engine.update_settings.assert_called_once_with(ready.settings)
        side_effects["save"].assert_called_once_with("ollama_model", "qwen3.5:4b-mlx")
        assert ready.model.status == "💾 ollama_model = qwen3.5:4b-mlx"

    def test_whisper_model_needs_restart(self, ready, side_effects):
        ready.update_setting("whisper_model", "small")
        side_effects["save"].assert_called_once_with("whisper_model", "small")
        assert "restart" in ready.model.status

    def test_empty_language_means_auto(self, ready, side_effects):
        ready.update_setting("language", "  ")
        assert ready.settings.language is None
        side_effects["save"].assert_called_once_with("language", None)
        assert ready.model.status == "💾 language = auto"

    def test_invalid_combination_is_rejected_before_saving(self, ready, engine, side_effects):
        ready.update_setting("language", "it")  # default model is base.en
        assert ready.settings.language is None
        engine.update_settings.assert_not_called()
        side_effects["save"].assert_not_called()
        assert ready.model.status.startswith("❌")

    def test_unknown_setting_is_rejected(self, ready, side_effects):
        ready.update_setting("nope", "x")
        side_effects["save"].assert_not_called()
        assert ready.model.status.startswith("❌")

    def test_hotkey_is_rebound_before_saving(self, ready, side_effects):
        ready.update_setting("hotkey", "<cmd>+<alt>+g")
        side_effects["rebind_hotkey"].assert_called_once_with("<cmd>+<alt>+g")
        side_effects["save"].assert_called_once_with("hotkey", "<cmd>+<alt>+g")
        assert ready.settings.hotkey == "<cmd>+<alt>+g"

    def test_invalid_hotkey_keeps_the_old_one(self, ready, side_effects):
        side_effects["rebind_hotkey"].side_effect = ValueError("bad combo")
        ready.update_setting("hotkey", "<nope>+g")
        side_effects["save"].assert_not_called()
        assert ready.settings.hotkey == HOTKEY
        assert ready.model.status == "❌ Invalid hotkey '<nope>+g': bad combo"

    def test_save_failure_still_applies_for_the_session(self, ready, engine, side_effects):
        side_effects["save"].side_effect = ConfigError("read-only")
        ready.update_setting("ollama_model", "x:1b")
        assert ready.settings.ollama_model == "x:1b"
        engine.update_settings.assert_called_once()
        assert (
            ready.model.status == "⚠️ ollama_model = x:1b applied for this session only: read-only"
        )

    def test_settings_survive_events(self, ready):
        ready.update_setting("ollama_model", "x:1b")
        model = apply(ready, StateChanged(State.IDLE, State.PROCESSING))
        assert model.settings.ollama_model == "x:1b"

    def test_before_attach_settings_still_change(self, presenter, side_effects):
        presenter.update_setting("ollama_model", "x:1b")
        assert presenter.settings.ollama_model == "x:1b"
        side_effects["save"].assert_called_once()


class TestPreview:
    def test_collapses_whitespace(self):
        assert preview("a\n\n  b\tc") == "a b c"

    def test_truncates_with_ellipsis(self):
        assert preview("abcdefghij", limit=5) == "abcd…"

    def test_short_text_untouched(self):
        assert preview("short", limit=5) == "short"


class TestShell:
    """Smoke test of the rumps wiring; skipped without the `gui` extra."""

    @pytest.fixture
    def app(self, presenter, tmp_path):
        rumps = pytest.importorskip("rumps")
        from localwhisper.gui import menubar

        on_quit = MagicMock()
        app = menubar.MenuBarApp(presenter, on_quit=on_quit, config_path=tmp_path / "c.toml")
        app.rumps, app.menubar, app.on_quit_mock = rumps, menubar, on_quit
        yield app
        app._timer.stop()

    def test_menu_layout(self, app):
        m = app.menubar
        assert list(app.menu.keys()) == [
            RECORD_TITLE,
            INSTRUCTION_TITLE,
            m.MODIFY_TEXT_TITLE,
            m.UNDO_TITLE,
            m.SHOW_TITLE,
            m.COPY_AGAIN_TITLE,
            "SeparatorMenuItem_1",
            LOADING_STATUS,
            "SeparatorMenuItem_2",
            m.SETTINGS_TITLE,
            m.QUIT_TITLE,
        ]
        assert app.title == LOADING_ICON
        for item in (app._record, app._instruction, app._modify_text, app._undo, app._show):
            assert item.callback is None, f"{item.title} must be disabled while loading"
        assert app._status.callback is None, "the status line is not clickable"

    def test_settings_submenu(self, app):
        titles = [item.title for item in app._setting_items.values()]
        assert titles[:4] == [
            f"Whisper model: {SETTINGS.whisper_model}…",
            f"Ollama model: {SETTINGS.ollama_model}…",
            "Language: auto…",
            f"Hotkey: {HOTKEY}…",
        ]
        assert app._auto_paste.state == 0

    def test_tick_applies_drained_model(self, app, presenter, engine):
        presenter.attach(engine)
        app._tick(app._timer)
        assert app.title == ICONS[State.IDLE]
        assert app._record.callback is not None
        assert app._modify_text.callback is None, "no text yet"
        assert HOTKEY in app._status.title

    def test_text_actions_enable_after_a_note(self, app, presenter, engine):
        presenter.attach(engine)
        presenter.post(RefinementReady("Hello.", instruction=None, fallback_raw=False))
        app._tick(app._timer)
        for item in (app._instruction, app._modify_text, app._undo, app._show, app._copy_again):
            assert item.callback is not None, item.title

    def test_record_item_flips_to_stop(self, app, presenter, engine):
        presenter.attach(engine)
        presenter.post(StateChanged(State.IDLE, State.RECORDING_NOTE))
        app._tick(app._timer)
        assert app._record.title == STOP_TITLE
        assert app.title == ICONS[State.RECORDING_NOTE]

    def test_auto_paste_click_toggles_and_checks(self, app, presenter, engine):
        presenter.attach(engine)
        app._tick(app._timer)
        app._auto_paste_clicked(app._auto_paste)
        app._tick(app._timer)
        assert presenter.settings.auto_paste is True
        assert app._auto_paste.state == 1

    def test_setting_window_saves_text(self, app, presenter, engine, monkeypatch):
        presenter.attach(engine)
        app._tick(app._timer)
        window = MagicMock()
        window.run.return_value = MagicMock(clicked=1, text="qwen3.5:4b-mlx")
        monkeypatch.setattr(app.rumps, "Window", MagicMock(return_value=window))
        app._setting_clicked("ollama_model")(None)
        app._tick(app._timer)
        assert presenter.settings.ollama_model == "qwen3.5:4b-mlx"
        assert app._setting_items["ollama_model"].title == "Ollama model: qwen3.5:4b-mlx…"

    def test_modify_window_sends_instruction(self, app, presenter, engine, monkeypatch):
        presenter.attach(engine)
        app._tick(app._timer)
        engine.last_refined_text = "Hello."
        window = MagicMock()
        window.run.return_value = MagicMock(clicked=1, text="shorter")
        monkeypatch.setattr(app.rumps, "Window", MagicMock(return_value=window))
        app._modify_text_clicked(None)
        engine.modify.assert_called_once_with("shorter")

    def test_modify_window_cancel_does_nothing(self, app, presenter, engine, monkeypatch):
        presenter.attach(engine)
        app._tick(app._timer)
        engine.last_refined_text = "Hello."
        window = MagicMock()
        window.run.return_value = MagicMock(clicked=0, text="shorter")
        monkeypatch.setattr(app.rumps, "Window", MagicMock(return_value=window))
        app._modify_text_clicked(None)
        engine.modify.assert_not_called()

    def test_quit_closes_before_terminating(self, app, monkeypatch):
        quit_application = MagicMock()
        monkeypatch.setattr(app.rumps, "quit_application", quit_application)
        app._quit_clicked(None)
        app.on_quit_mock.assert_called_once_with()
        quit_application.assert_called_once_with()


class TestActivateAndRun:
    """Regression tests for the "can't type in the popup" bug.

    Clicking a status-bar item does not make the process frontmost, so a modal shown
    later can appear on screen while keystrokes keep going to whatever app *was*
    frontmost (github.com/jaredks/rumps/issues/127). `_activate_and_run` reactivates
    before showing, and returns focus afterwards so LocalWhisper doesn't linger as the
    active app and steal the next auto-paste.
    """

    @pytest.fixture
    def app(self, presenter, tmp_path, monkeypatch):
        pytest.importorskip("rumps")
        from localwhisper.gui import menubar

        other_app = MagicMock()
        other_app.processIdentifier.return_value = 999999  # never our own pid
        workspace = MagicMock()
        workspace.frontmostApplication.return_value = other_app
        nsapp = MagicMock()

        # Replace the module-level names, not a method on the real PyObjC classes:
        # PyObjC validates method signatures on the actual bridged class and rejects
        # a plain Python callable there.
        workspace_class = MagicMock(sharedWorkspace=MagicMock(return_value=workspace))
        application_class = MagicMock(sharedApplication=MagicMock(return_value=nsapp))
        monkeypatch.setattr(menubar, "NSWorkspace", workspace_class)
        monkeypatch.setattr(menubar, "NSApplication", application_class)

        app = menubar.MenuBarApp(presenter, on_quit=MagicMock(), config_path=tmp_path / "c.toml")
        app.other_app, app.nsapp, app.workspace = other_app, nsapp, workspace
        app.rumps = pytest.importorskip("rumps")
        yield app
        app._timer.stop()

    def test_activates_shows_then_restores_focus(self, app):
        show = MagicMock(return_value="result")
        order = []
        app.nsapp.activateIgnoringOtherApps_.side_effect = lambda _: order.append("activate")
        show.side_effect = lambda: order.append("show") or "result"
        app.other_app.activateWithOptions_.side_effect = lambda _: order.append("restore")

        result = app._activate_and_run(show)

        assert result == "result"
        app.nsapp.activateIgnoringOtherApps_.assert_called_once_with(True)
        app.other_app.activateWithOptions_.assert_called_once_with(0)
        assert order == ["activate", "show", "restore"]

    def test_restores_focus_even_if_show_raises(self, app):
        with pytest.raises(RuntimeError):
            app._activate_and_run(MagicMock(side_effect=RuntimeError("boom")))
        app.other_app.activateWithOptions_.assert_called_once_with(0)

    def test_does_not_reactivate_itself(self, app):
        """If we were already frontmost, restoring focus to "ourselves" is a no-op."""
        app.other_app.processIdentifier.return_value = os.getpid()
        app._activate_and_run(MagicMock(return_value=None))
        app.other_app.activateWithOptions_.assert_not_called()

    def test_no_previous_app_is_handled(self, app):
        app.workspace.frontmostApplication.return_value = None
        app._activate_and_run(MagicMock(return_value=None))  # must not raise

    def test_show_alert_goes_through_activation(self, app, monkeypatch):
        alert = MagicMock()
        monkeypatch.setattr(app.rumps, "alert", alert)
        app.show_alert("LocalWhisper error", "Refinement failed")
        app.nsapp.activateIgnoringOtherApps_.assert_called_once_with(True)
        alert.assert_called_once_with(title="LocalWhisper error", message="Refinement failed")
        app.other_app.activateWithOptions_.assert_called_once_with(0)

    def test_modify_window_goes_through_activation(self, app, presenter, engine, monkeypatch):
        presenter.attach(engine)
        app._tick(app._timer)
        engine.last_refined_text = "Hello."
        window = MagicMock()
        window.run.return_value = MagicMock(clicked=0, text="")
        monkeypatch.setattr(app.rumps, "Window", MagicMock(return_value=window))

        app._modify_text_clicked(None)

        app.nsapp.activateIgnoringOtherApps_.assert_called_once_with(True)
        app.other_app.activateWithOptions_.assert_called_once_with(0)
