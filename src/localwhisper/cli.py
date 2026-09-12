"""Interactive terminal presenter for LocalWhisper.

This module owns everything that is specific to running in a terminal: the menu,
`input()`, emoji status lines, the clipboard and the global hotkey binding. All
recording and AI state lives in `engine.Engine`; this class only translates its
events into text and its menu choices into engine calls.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

import pyperclip
from pynput import keyboard

from .audio import AudioRecorder
from .config import ConfigError, Settings, load_settings
from .core import AIProcessor
from .engine import Engine
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


class LocalWhisperCLI:
    """Menu loop and event printer over an `Engine`."""

    def __init__(self, engine: Engine, hotkey: str = Settings.hotkey) -> None:
        self.engine = engine
        self.hotkey = hotkey
        self.engine.subscribe(self.on_event)
        self.menu_active: bool = True

    # ----------------------------------------------------------------- output

    def print_menu(self) -> None:
        """Display the interactive menu."""
        print("\n--------------------------------")
        print(f"[r] 🎤 Record New (HotKey: {self.hotkey})")
        print("[v] 🗣️  Modify with Voice (ENTER to stop)")
        print("[m] ✏️  Modify with Text")
        print("[u] ↩️  Undo last edit")
        print("[s] 📋 Show last text")
        print("[q] 🚪 Quit")
        print("> ", end="", flush=True)

    def on_event(self, event: Event) -> None:
        """Render one engine event. Runs on the engine's worker or hotkey thread."""
        match event:
            case StateChanged(current=State.IDLE):
                self.print_menu()
            case RecordingStarted(kind=RecordingKind.NOTE):
                print("\n\n🎤 Recording... (Press ENTER or hotkey again to stop)")
            case RecordingStarted(kind=RecordingKind.INSTRUCTION):
                print("\n\n🎤 Recording instruction... (Press ENTER to stop)")
            case RecordingStopped(seconds=seconds, dropped_chunks=dropped):
                print(f"\n🛑 Recording stopped ({seconds:.1f}s).")
                if dropped:
                    print(f"⚠️ Input overflow: {dropped} chunk(s) dropped during recording.")
            case Transcribing(kind=RecordingKind.NOTE):
                print("📝 Transcribing...")
            case Transcribing(kind=RecordingKind.INSTRUCTION):
                print("📝 Transcribing instruction...")
            case TranscriptReady(kind=RecordingKind.INSTRUCTION, text=text):
                print(f'🗣️  Instruction: "{text}"')
            case TranscriptReady(seconds=seconds):
                print(f"✅ Transcribed in {seconds:.1f}s")
            case Refining(instruction=None):
                print("🧠 Refining...")
            case Refining(instruction=instruction):
                print(f"🧠 Refining with instruction: '{instruction}'...")
            case RefinementReady(text=text, fallback_raw=True):
                print("📋 Copying the RAW transcript to the clipboard instead.")
                self._copy_to_clipboard(text)
            case RefinementReady(text=text, instruction=None):
                print(f"\n✨ Final Output:\n{text}\n")
                self._copy_to_clipboard(text)
            case RefinementReady(text=text):
                print(f"\n✨ New Output:\n{text}\n")
                self._copy_to_clipboard(text)
            case Reverted(text=text, revisions_left=left):
                label = "raw transcript" if left == 0 else f"revision {left}"
                print(f"\n↩️  Reverted to {label}:\n{text}\n")
                self._copy_to_clipboard(text)
            case Notice(message=message):
                print(f"⚠️ {message}")
            case Error(message=message):
                print(f"❌ {message}")
            case _:
                pass

    def _copy_to_clipboard(self, text: str) -> None:
        """Copy text to the system clipboard, reporting failure instead of raising."""
        try:
            pyperclip.copy(text)
            print("📋 Copied to clipboard!")
        except pyperclip.PyperclipException as e:
            print(f"❌ Could not copy to clipboard: {e}")

    # ---------------------------------------------------------------- commands

    def on_activate(self) -> None:
        """Global hotkey handler: toggle a note recording."""
        self.engine.toggle()

    def record_instruction(self) -> None:
        """Menu `[v]`: record a spoken instruction, stopped by ENTER only."""
        if not self.engine.last_raw_text:
            print("\n❌ No text to modify yet! Record something first.")
            self.print_menu()
            return
        if not self.engine.start(RecordingKind.INSTRUCTION):
            print("\n❌ Already recording.")
            self.print_menu()
            return
        input()
        self.engine.stop()

    def modify_last_text(self) -> None:
        """Menu `[m]`: type an instruction; the refinement runs on the engine worker."""
        if not self.engine.last_raw_text:
            print("\n❌ No text to modify yet! Record something first.")
            return

        print(f'\nOriginal Raw Text: "{self.engine.last_raw_text}"')
        print(f'Current Text: "{self.engine.last_refined_text}"')

        instruction = input("\n✏️  Enter modification instruction: ")
        self.engine.modify(instruction)

    def show_last_text(self) -> None:
        """Menu `[s]`: current text plus the chain of instructions that produced it."""
        text = self.engine.last_refined_text
        if not text:
            print("\nNo text recorded yet.")
            return
        history = self.engine.history
        applied = [r.instruction for r in history if r.instruction]
        if applied:
            print(f"\nLast Text (after {len(history)} edit(s): {' → '.join(applied)}):\n{text}")
        else:
            print(f"\nLast Text:\n{text}")

    def show_menu(self) -> None:
        """Run the menu loop until the user quits or stdin closes."""
        self.print_menu()

        while self.menu_active:
            try:
                choice = input().strip().lower()
            except EOFError:
                break

            if not choice:
                # Bare ENTER stops a hotkey/`r` recording; an instruction recording
                # is stopped by the `input()` inside record_instruction().
                if self.engine.state is State.RECORDING_NOTE:
                    self.engine.stop()
                continue

            if choice == "r":
                if not self.engine.toggle():
                    self.print_menu()
            elif choice == "v":
                self.record_instruction()
            elif choice == "m":
                self.modify_last_text()
                if not self.engine.busy:
                    self.print_menu()
            elif choice == "u":
                self.engine.undo()
                self.print_menu()
            elif choice == "s":
                self.show_last_text()
                self.print_menu()
            elif choice == "q":
                self.menu_active = False
                print("Byee! 👋")
            else:
                print("Unknown command. Try 'r', 'v', 'm', 'u', 's', or 'q'.")
                self.print_menu()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `localwhisper` console script."""
    try:
        settings = load_settings(argv)
    except ConfigError as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        return 2

    if settings.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
        )

    print("\n=== LocalWhisper CLI ===")
    print(f"🎧 Loading Whisper model: {settings.whisper_model}... please wait.")
    try:
        ai = AIProcessor(settings)
    except Exception as e:  # noqa: BLE001 - startup: any load failure is fatal and reported
        print(f"❌ Could not load Whisper model: {e}")
        return 1
    print("✅ Whisper model loaded.")

    warning = ai.check_ollama()
    if warning:
        print(f"⚠️ {warning}")
    else:
        print(f"🧠 Ollama model ready: {settings.ollama_model}")

    recorder = AudioRecorder()
    engine = Engine(recorder, ai)
    app = LocalWhisperCLI(engine, hotkey=settings.hotkey)

    listener = keyboard.GlobalHotKeys({settings.hotkey: app.on_activate})
    listener.daemon = True
    listener.start()
    print("🚀 Live Capture Ready!")

    try:
        app.show_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        listener.stop()
        if not engine.shutdown():
            print("⚠️ A job was still running; exiting without waiting for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
