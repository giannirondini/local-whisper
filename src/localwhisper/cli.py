"""Main entry point for LocalWhisper CLI application.

This module provides the interactive command-line interface,
global hotkey handling, and orchestration of audio recording
and AI processing.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import pyperclip
from pynput import keyboard

# Import LocalWhisper Modules
from .core import AIProcessor
from .audio import AudioRecorder

# --- CONFIGURATION ---
HOTKEY_COMBINATION: str = "<cmd>+<shift>+g"


class LocalWhisperCLI:
    """Interactive CLI for LocalWhisper voice-to-text application."""

    def __init__(self) -> None:
        """Initialize the CLI application."""
        self.recorder: AudioRecorder = AudioRecorder()
        self.ai: AIProcessor = AIProcessor()  # Initialize AI (loads Whisper model)

        self.is_recording: bool = False
        self.processing_lock: threading.Lock = threading.Lock()

        # State
        self.last_raw_text: Optional[str] = None
        self.last_refined_text: Optional[str] = None
        self.menu_active: bool = True
        self.next_recording_is_instruction: bool = False
        self.hotkey_locked: bool = False  # True when the menu owns the recording

    def print_menu(self) -> None:
        """Display the interactive menu."""
        print("\n--------------------------------")
        print(f"[r] 🎤 Record New (HotKey: {HOTKEY_COMBINATION})")
        print("[v] 🗣️  Modify with Voice (ENTER to stop)")
        print("[m] ✏️  Modify with Text")
        print("[s] 📋 Show last text")
        print("[q] 🚪 Quit")
        print("> ", end="", flush=True)

    def on_activate(self) -> None:
        """Handle hotkey activation to toggle recording."""
        if self.hotkey_locked:
            return  # Menu owns this recording; hotkey has no effect
        with self.processing_lock:
            if self.is_recording:
                self.stop_and_process()
            else:
                self.start_capture()

    def start_capture(self) -> None:
        """Start audio capture."""
        print("\n\n🎤 Starting recording... (Press ENTER or hotkey again to stop)")
        self.is_recording = True
        self.recorder.start_recording()

    def stop_and_process(self) -> None:
        """Stop recording and process the captured audio."""
        print("\n\n🛑 Stopping recording...")
        self.is_recording = False
        wav_file = self.recorder.stop_recording()
        
        if not wav_file:
            print("❌ No audio recorded.")
            return

        # Process in a separate thread to not block the listener
        if self.next_recording_is_instruction:
             self.next_recording_is_instruction = False # Reset flag
             threading.Thread(target=self._process_instruction, args=(wav_file,)).start()
        else:
            threading.Thread(target=self._process_audio, args=(wav_file,)).start()

    def _process_audio(self, wav_file: str) -> None:
        """Process recorded audio: transcribe, refine, and copy to clipboard.

        Args:
            wav_file: Path to the recorded WAV file.
        """
        try:
            # 1. Transcribe
            print(f"\n📝 Transcribing...")
            raw_text = self.ai.transcribe(wav_file)
            self.last_raw_text = raw_text 
            
            if not raw_text:
                print("⚠️ No speech detected.")
                return

            # 2. Refine
            print(f"🧠 Refining...")
            refined_text = self.ai.refine_text(raw_text)
            self.last_refined_text = refined_text 
            
            print(f"\n✨ Final Output:\n{refined_text}\n")

            # 3. Copy to Clipboard
            pyperclip.copy(refined_text)
            print("📋 Copied to clipboard!")

            # 4. Cleanup
            if os.path.exists(wav_file):
                os.remove(wav_file)
            
            self.print_menu()

        except Exception as e:
            print(f"❌ Error in processing: {e}")
            if os.path.exists(wav_file):
                os.remove(wav_file)
            self.print_menu()

    def _process_instruction(self, wav_file: str) -> None:
        """Process a voice instruction to modify the last transcription.

        Args:
            wav_file: Path to the recorded instruction WAV file.
        """
        try:
            if not self.last_raw_text:
                print("\n⚠️ No previous text to modify. Recording treated as new transcription.")
                self._process_audio(wav_file)
                return

            print(f"\n📝 Transcribing Instruction...")
            instruction_text = self.ai.transcribe(wav_file)
            
            if not instruction_text:
                print("⚠️ No instruction detected.")
                os.remove(wav_file)
                self.print_menu()
                return

            print(f"🗣️  Instruction: \"{instruction_text}\"")
            print(f"🧠 Refining with instruction...")
            
            new_text = self.ai.refine_text(self.last_raw_text, instruction=instruction_text)
            self.last_refined_text = new_text
            
            print(f"\n✨ New Output:\n{new_text}\n")
            pyperclip.copy(new_text)
            print("📋 Copied to clipboard!")
            
            os.remove(wav_file)
            self.print_menu()

        except Exception as e:
            print(f"❌ Error during instruction processing: {e}")
            if os.path.exists(wav_file):
                os.remove(wav_file)
            self.print_menu()

    def modify_last_text(self) -> None:
        """Interactively modify the last transcription with a text instruction."""
        if not self.last_raw_text:
            print("\n❌ No text to modify yet! Record something first.")
            return

        print(f"\nOriginal Raw Text: \"{self.last_raw_text}\"")
        print(f"Current Refined Text: \"{self.last_refined_text}\"")
        
        instruction = input("\n✏️  Enter modification instruction: ")
        if not instruction.strip():
            return

        print(f"\n🧠 Re-refining with instruction: '{instruction}'...")
        try:
            new_text = self.ai.refine_text(self.last_raw_text, instruction=instruction)
            self.last_refined_text = new_text
            print(f"\n✨ New Output:\n{new_text}\n")
            pyperclip.copy(new_text)
            print("📋 Copied to clipboard!")
        except Exception as e:
            print(f"❌ Error modifying text: {e}")

    def show_menu(self) -> None:
        """Display the menu and handle user input."""
        print("\n=== LocalWhisper CLI ===")
        print("Initializing AI... please wait.")
        # Trigger model load if not already done (it is done in __init__)
        
        self.print_menu()

        while self.menu_active:
            try:
                choice = input().strip().lower()
            except EOFError:
                break
            
            if not choice:
                if self.is_recording and not self.hotkey_locked:
                    self.stop_and_process()
                continue

            if choice == 'r':
                self.on_activate()
            elif choice == 'v':
                if not self.last_raw_text:
                    print("\n❌ No text to modify yet! Record something first.")
                    self.print_menu()
                    continue
                self.next_recording_is_instruction = True
                self.hotkey_locked = True
                self.start_capture()
                print("\nPress ENTER to stop recording...")
                input()
                self.hotkey_locked = False
                self.stop_and_process()
            elif choice == 'm':
                self.modify_last_text()
            elif choice == 's':
                if self.last_refined_text:
                    print(f"\nLast Text:\n{self.last_refined_text}")
                else:
                    print("\nNo text recorded yet.")
            elif choice == 'q':
                self.menu_active = False
                print("Byee! 👋")
                os._exit(0)
            else:
                print("Unknown command. Try 'r', 'v', 'm', 's', or 'q'.")
            
            if choice not in ["r", "v"]:
                self.print_menu()


def main() -> None:
    """Main entry point for LocalWhisper."""
    # ctranslate2 (used by faster-whisper) registers semaphores with
    # multiprocessing.resource_tracker that are never cleanly released when
    # os._exit() is called.  Suppress the resulting UserWarning by setting
    # PYTHONWARNINGS before the resource tracker subprocess is forked.
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::UserWarning:multiprocessing.resource_tracker",
    )

    print("🚀 Live Capture Ready!")
    try:
        app = LocalWhisperCLI()
        
        # Setup Global Hotkey
        listener = keyboard.GlobalHotKeys({
            HOTKEY_COMBINATION: app.on_activate
        })
        listener.daemon = True
        listener.start()
        
        # Start Interactive CLI
        app.show_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
        os._exit(0)

if __name__ == "__main__":
    main()
