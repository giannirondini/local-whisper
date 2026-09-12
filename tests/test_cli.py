"""Tests for the terminal presenter (review sessions S1/S3).

The engine is a mock: these tests check that menu choices become the right engine
calls and that engine events become the right output and clipboard writes. The
behaviour that used to live here (F-01/F-02/F-16 regressions) is now tested against
the engine in `test_engine.py`.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from localwhisper.cli import LocalWhisperCLI, main
from localwhisper.config import ConfigError, Settings
from localwhisper.engine import Engine, Revision
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


@pytest.fixture
def engine():
    eng = MagicMock(spec=Engine)
    eng.state = State.IDLE
    eng.busy = False
    eng.last_raw_text = None
    eng.last_refined_text = None
    eng.history = ()
    return eng


@pytest.fixture
def cli(engine):
    with patch("localwhisper.cli.pyperclip") as clipboard:
        app = LocalWhisperCLI(engine)
        app.clipboard = clipboard
        yield app


def feed(cli, *choices):
    """Run the menu loop over a scripted sequence of inputs, then EOF."""
    with patch("builtins.input", side_effect=[*choices, EOFError]):
        cli.show_menu()


class TestEvents:
    def test_subscribes_on_construction(self, cli, engine):
        engine.subscribe.assert_called_once_with(cli.on_event)

    def test_refinement_ready_prints_and_copies(self, cli, capsys):
        cli.on_event(RefinementReady("Hello.", instruction=None, fallback_raw=False))

        assert "✨ Final Output:\nHello." in capsys.readouterr().out
        cli.clipboard.copy.assert_called_once_with("Hello.")

    def test_instruction_result_prints_new_output(self, cli, capsys):
        cli.on_event(RefinementReady("- Hello.", instruction="list", fallback_raw=False))

        assert "✨ New Output:\n- Hello." in capsys.readouterr().out
        cli.clipboard.copy.assert_called_once_with("- Hello.")

    def test_fallback_copies_raw_and_says_so(self, cli, capsys):
        """F-02: the raw transcript is copied and the user is told why."""
        cli.on_event(Error("Refinement failed: Cannot reach Ollama"))
        cli.on_event(RefinementReady("raw words", instruction=None, fallback_raw=True))

        out = capsys.readouterr().out
        assert "❌ Refinement failed: Cannot reach Ollama" in out
        assert "RAW transcript" in out
        cli.clipboard.copy.assert_called_once_with("raw words")

    def test_clipboard_failure_is_reported_not_raised(self, cli, capsys):
        cli.clipboard.PyperclipException = Exception
        cli.clipboard.copy.side_effect = Exception("no clipboard")

        cli.on_event(RefinementReady("x", instruction=None, fallback_raw=False))

        assert "Could not copy to clipboard" in capsys.readouterr().out

    def test_menu_is_reprinted_when_engine_returns_to_idle(self, cli, capsys):
        cli.on_event(StateChanged(State.PROCESSING, State.IDLE))
        assert "[r] 🎤 Record New" in capsys.readouterr().out

        cli.on_event(StateChanged(State.IDLE, State.RECORDING_NOTE))
        assert "[r]" not in capsys.readouterr().out

    def test_recording_events(self, cli, capsys):
        cli.on_event(RecordingStarted(RecordingKind.NOTE))
        cli.on_event(RecordingStopped(RecordingKind.NOTE, 2.5, dropped_chunks=2))

        out = capsys.readouterr().out
        assert "🎤 Recording..." in out
        assert "🛑 Recording stopped (2.5s)" in out
        assert "2 chunk(s) dropped" in out

    def test_instruction_transcript_is_echoed(self, cli, capsys):
        cli.on_event(TranscriptReady(RecordingKind.INSTRUCTION, "make it short", 0.4))
        assert 'Instruction: "make it short"' in capsys.readouterr().out

    def test_reverted_prints_and_copies(self, cli, capsys):
        cli.on_event(Reverted("Hello, world.", revisions_left=1))
        assert "Reverted to revision 1:\nHello, world." in capsys.readouterr().out
        cli.clipboard.copy.assert_called_once_with("Hello, world.")

        cli.on_event(Reverted("hello world", revisions_left=0))
        assert "Reverted to raw transcript" in capsys.readouterr().out

    def test_notice_and_error_prefixes(self, cli, capsys):
        cli.on_event(Notice("No speech detected."))
        cli.on_event(Error("Transcription failed: boom"))

        out = capsys.readouterr().out
        assert "⚠️ No speech detected." in out
        assert "❌ Transcription failed: boom" in out


class TestMenu:
    def test_hotkey_toggles(self, cli, engine):
        cli.on_activate()
        engine.toggle.assert_called_once_with()

    def test_r_toggles(self, cli, engine):
        engine.toggle.return_value = True
        feed(cli, "r")
        engine.toggle.assert_called_once_with()

    def test_bare_enter_stops_a_note_recording_only(self, cli, engine):
        engine.state = State.RECORDING_NOTE
        feed(cli, "")
        engine.stop.assert_called_once_with()

        engine.stop.reset_mock()
        engine.state = State.RECORDING_INSTRUCTION
        feed(cli, "")
        engine.stop.assert_not_called()

    def test_v_records_instruction_until_enter(self, cli, engine):
        engine.last_raw_text = "note"
        engine.start.return_value = True

        feed(cli, "v", "")  # "v", then the ENTER that stops the instruction

        engine.start.assert_called_once_with(RecordingKind.INSTRUCTION)
        engine.stop.assert_called_once_with()

    def test_v_without_text_does_not_record(self, cli, engine, capsys):
        feed(cli, "v")

        engine.start.assert_not_called()
        assert "No text to modify yet" in capsys.readouterr().out

    def test_m_queues_instruction_on_engine(self, cli, engine, capsys):
        """F-17: the CLI hands the instruction to the engine and returns to the menu."""
        engine.last_raw_text = "note"
        engine.last_refined_text = "Note."
        engine.modify.return_value = True

        feed(cli, "m", "shorten it")

        engine.modify.assert_called_once_with("shorten it")
        out = capsys.readouterr().out
        assert 'Original Raw Text: "note"' in out
        assert 'Current Text: "Note."' in out

    def test_m_without_text_does_not_prompt(self, cli, engine, capsys):
        feed(cli, "m")

        engine.modify.assert_not_called()
        assert "No text to modify yet" in capsys.readouterr().out

    def test_s_shows_last_text(self, cli, engine, capsys):
        engine.last_refined_text = "Final."
        engine.history = (Revision(None, "Final."),)
        feed(cli, "s")
        assert "Last Text:\nFinal." in capsys.readouterr().out

    def test_s_shows_applied_instructions(self, cli, engine, capsys):
        engine.last_refined_text = "- Final."
        engine.history = (Revision(None, "Final."), Revision("list", "- Final."))
        feed(cli, "s")
        out = capsys.readouterr().out
        assert "after 2 edit(s): list" in out
        assert "- Final." in out

    def test_u_undoes_and_reprints_menu(self, cli, engine, capsys):
        feed(cli, "u")
        engine.undo.assert_called_once_with()
        assert capsys.readouterr().out.count("[u] ↩️  Undo last edit") == 2

    def test_menu_shows_configured_hotkey(self, engine, capsys):
        with patch("localwhisper.cli.pyperclip"):
            LocalWhisperCLI(engine, hotkey="<ctrl>+<alt>+r").print_menu()
        assert "HotKey: <ctrl>+<alt>+r" in capsys.readouterr().out

    def test_q_ends_loop_without_calling_exit(self, cli, engine):
        """F-11: quitting returns from the loop; shutdown belongs to main()."""
        with patch("builtins.input", side_effect=["q", "r"]):
            cli.show_menu()

        assert cli.menu_active is False
        engine.toggle.assert_not_called()  # loop stopped before the second input

    def test_unknown_command_reprints_menu(self, cli, engine, capsys):
        feed(cli, "zzz")
        out = capsys.readouterr().out
        assert "Unknown command" in out
        assert out.count("[r] 🎤 Record New") == 2  # initial + after the error


@pytest.fixture
def wired_main():
    """`main()` with settings, AI, recorder, engine and hotkeys all mocked."""
    settings = Settings(hotkey="<ctrl>+x", ollama_model="my-model")
    with (
        patch("localwhisper.cli.load_settings", return_value=settings) as load,
        patch("localwhisper.cli.AIProcessor") as ai_cls,
        patch("localwhisper.cli.AudioRecorder"),
        patch("localwhisper.cli.Engine") as engine_cls,
        patch("localwhisper.cli.keyboard.GlobalHotKeys") as hotkeys,
        patch("builtins.input", side_effect=["q"]),
    ):
        ai_cls.return_value.check_ollama.return_value = None
        engine_cls.return_value.shutdown.return_value = True
        engine_cls.return_value.busy = False
        yield {
            "settings": settings,
            "load": load,
            "ai_cls": ai_cls,
            "engine_cls": engine_cls,
            "hotkeys": hotkeys,
        }


class TestMain:
    def test_main_shuts_engine_down_after_menu(self, wired_main):
        """F-11: normal exit path, no os._exit()."""
        assert main([]) == 0

        wired_main["engine_cls"].return_value.shutdown.assert_called_once_with()
        assert wired_main["hotkeys"].return_value.method_calls[-1] == call.stop()

    def test_main_wires_settings_through(self, wired_main, capsys):
        """F-12: settings reach the AI processor, the hotkey binding and the menu."""
        main(["--hotkey", "<ctrl>+x"])

        wired_main["load"].assert_called_once_with(["--hotkey", "<ctrl>+x"])
        wired_main["ai_cls"].assert_called_once_with(wired_main["settings"])
        bindings = wired_main["hotkeys"].call_args.args[0]
        assert list(bindings) == ["<ctrl>+x"]
        out = capsys.readouterr().out
        assert "HotKey: <ctrl>+x" in out
        assert "Ollama model ready: my-model" in out

    def test_main_prints_ollama_warning(self, wired_main, capsys):
        """F-20: a missing model is reported at startup, not on the first note."""
        wired_main["ai_cls"].return_value.check_ollama.return_value = "Model 'x' is not installed"

        assert main([]) == 0

        assert "⚠️ Model 'x' is not installed" in capsys.readouterr().out

    def test_main_reports_config_error(self, capsys):
        with patch("localwhisper.cli.load_settings", side_effect=ConfigError("bad url")):
            assert main([]) == 2
        assert "Configuration error: bad url" in capsys.readouterr().err

    def test_main_reports_model_load_failure(self, capsys):
        with (
            patch("localwhisper.cli.load_settings", return_value=Settings()),
            patch("localwhisper.cli.AIProcessor", side_effect=RuntimeError("no model")),
        ):
            assert main([]) == 1

        assert "Could not load Whisper model: no model" in capsys.readouterr().out
