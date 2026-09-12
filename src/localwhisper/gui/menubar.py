"""macOS menu bar app: the `rumps` shell around `MenuBarPresenter`.

Entry point of the `localwhisper-gui` console script. This module is the only
place that imports `rumps`; everything it does is (1) build the status item and
menu, (2) drain the presenter's queue from a main-thread timer and apply the
resulting `MenuModel`, (3) open the small `rumps.Window`/`rumps.alert` dialogs,
(4) bind the global hotkey and synthesize Cmd+V for auto-paste, and (5) shut the
engine down before the process terminates.

The Whisper model is loaded on a background thread so the status item appears
immediately with a "loading" glyph instead of the app looking dead for a few
seconds (or minutes, on the first-run download).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import rumps
from AppKit import NSApplication, NSWorkspace
from pynput import keyboard

from ..audio import AudioRecorder
from ..config import (
    ConfigError,
    Settings,
    build_parser,
    load_settings,
    resolve_config_path,
    save_setting,
)
from ..core import AIProcessor
from ..engine import Engine
from .notify import notify
from .presenter import MenuBarPresenter, MenuModel, preview

logger = logging.getLogger(__name__)

T = TypeVar("T")

APP_NAME = "LocalWhisper"
MODIFY_TEXT_TITLE = "Modify with Text…"
UNDO_TITLE = "Undo Last Edit"
SHOW_TITLE = "Show Last Text…"
COPY_AGAIN_TITLE = "Copy Again"
SETTINGS_TITLE = "Settings"
AUTO_PASTE_TITLE = "Auto-paste (Cmd+V after copying)"
QUIT_TITLE = "Quit LocalWhisper"
DRAIN_INTERVAL = 0.1
"""Seconds between queue drains; the latency between an engine event and the menu."""

SETTING_LABELS: dict[str, str] = {
    "whisper_model": "Whisper model",
    "ollama_model": "Ollama model",
    "language": "Language",
    "hotkey": "Hotkey",
}
SETTING_HINTS: dict[str, str] = {
    "whisper_model": "faster-whisper model name (base.en, base, small, medium, large-v3). "
    "Takes effect on the next launch.",
    "ollama_model": "Ollama model tag, as in `ollama list`. Applies immediately.",
    "language": "ISO 639-1 code (en, it, …), or empty for auto-detect. Applies immediately.",
    "hotkey": "pynput syntax, e.g. <cmd>+<shift>+g. Applies immediately.",
}


class MenuBarApp(rumps.App):  # type: ignore[misc]  # rumps has no type stubs
    """Status item + menu. All methods run on the main thread (rumps guarantees it)."""

    def __init__(
        self, presenter: MenuBarPresenter, on_quit: Callable[[], None], config_path: Path
    ) -> None:
        super().__init__(APP_NAME, title=presenter.model.icon, quit_button=None)
        self._presenter = presenter
        self._on_quit = on_quit

        self._record = rumps.MenuItem(presenter.model.record_title, callback=self._record_clicked)
        self._instruction = rumps.MenuItem(
            presenter.model.instruction_title, callback=self._instruction_clicked
        )
        self._modify_text = rumps.MenuItem(MODIFY_TEXT_TITLE, callback=self._modify_text_clicked)
        self._undo = rumps.MenuItem(UNDO_TITLE, callback=self._undo_clicked)
        self._show = rumps.MenuItem(SHOW_TITLE, callback=self._show_clicked)
        self._copy_again = rumps.MenuItem(COPY_AGAIN_TITLE, callback=self._copy_again_clicked)
        self._status = rumps.MenuItem(presenter.model.status)  # no callback: greyed out

        self._setting_items = {
            name: rumps.MenuItem(label, callback=self._setting_clicked(name))
            for name, label in SETTING_LABELS.items()
        }
        self._auto_paste = rumps.MenuItem(AUTO_PASTE_TITLE, callback=self._auto_paste_clicked)
        settings = rumps.MenuItem(SETTINGS_TITLE)
        settings.update(
            [
                *self._setting_items.values(),
                self._auto_paste,
                None,
                rumps.MenuItem(f"Config file: {_pretty(config_path)}"),  # label only
            ]
        )

        self.menu = [
            self._record,
            self._instruction,
            self._modify_text,
            self._undo,
            self._show,
            self._copy_again,
            None,
            self._status,
            None,
            settings,
            rumps.MenuItem(QUIT_TITLE, callback=self._quit_clicked),
        ]
        self.apply(presenter.model)
        # Created on the main thread, so the NSTimer joins the main run loop.
        self._timer = rumps.Timer(self._tick, DRAIN_INTERVAL)
        self._timer.start()

    # ------------------------------------------------------------ rendering

    def apply(self, model: MenuModel) -> None:
        self.title = model.icon
        self._status.title = model.status
        self._record.title = model.record_title
        self._instruction.title = model.instruction_title
        _enable(self._record, self._record_clicked, model.record_enabled)
        _enable(self._instruction, self._instruction_clicked, model.instruction_enabled)
        for item, callback in (
            (self._modify_text, self._modify_text_clicked),
            (self._undo, self._undo_clicked),
            (self._show, self._show_clicked),
            (self._copy_again, self._copy_again_clicked),
        ):
            _enable(item, callback, model.text_actions_enabled)
        for name, item in self._setting_items.items():
            value = getattr(model.settings, name)
            item.title = f"{SETTING_LABELS[name]}: {'auto' if value is None else value}…"
        self._auto_paste.state = 1 if model.settings.auto_paste else 0

    def _tick(self, _timer: rumps.Timer) -> None:
        model = self._presenter.drain()
        if model is not None:
            self.apply(model)

    # --------------------------------------------------------------- modals

    def _activate_and_run(self, show: Callable[[], T]) -> T:
        """Reactivate this app before showing a modal, then hand focus back.

        Clicking a status-bar item does not make the process the frontmost
        application. A `rumps.Window`/`rumps.alert` shown later can be visually
        in front while keystrokes still go to whichever app *is* frontmost,
        silently blocking typing (https://github.com/jaredks/rumps/issues/127).

        KNOWN LIMITATION: on recent macOS (confirmed on the Darwin 25.x line;
        see Apple Developer Forums threads 739075, 739524, 807805), the OS's
        focus-stealing prevention can make `activateIgnoringOtherApps_` (and
        even the deprecated Carbon `SetFrontProcessWithOptions` fallback below)
        silently do nothing for a process not launched from a proper `.app`
        bundle. We still call every documented lever, best-effort, but a
        packaged `.app` (UI plan phase 3) is the actual fix, not this method.
        """
        previous = NSWorkspace.sharedWorkspace().frontmostApplication()
        _force_activate()
        try:
            return show()
        finally:
            if previous is not None and previous.processIdentifier() != os.getpid():
                previous.activateWithOptions_(0)

    def show_alert(self, title: str, message: str) -> None:
        """A blocking, impossible-to-miss dialog for failures (see `Error` in the presenter).

        Notifications can be silently dropped (no bundle id yet: see `notify.py`) and the
        status line is only visible when the menu is open, so `Error` needs a louder channel.
        """
        self._activate_and_run(lambda: rumps.alert(title=title, message=message))

    # ------------------------------------------------------------- actions

    def _record_clicked(self, _item: rumps.MenuItem) -> None:
        self._presenter.toggle_record()

    def _instruction_clicked(self, _item: rumps.MenuItem) -> None:
        self._presenter.toggle_instruction()

    def _modify_text_clicked(self, _item: rumps.MenuItem) -> None:
        found = self._presenter.last_text()
        if found is None:
            return
        response = self._activate_and_run(
            lambda: rumps.Window(
                title="Modify with Text",
                message=f"Instruction for the current text:\n{preview(found[0], 200)}",
                ok="Apply",
                cancel="Cancel",
                dimensions=(360, 48),
            ).run()
        )
        if response.clicked and response.text.strip():
            self._presenter.modify(response.text)

    def _undo_clicked(self, _item: rumps.MenuItem) -> None:
        self._presenter.undo()

    def _show_clicked(self, _item: rumps.MenuItem) -> None:
        found = self._presenter.last_text()
        if found is None:
            return
        text, applied = found
        message = f"After {len(applied)} edit(s): {' → '.join(applied)}" if applied else "Refined"
        response = self._activate_and_run(
            lambda: rumps.Window(
                title="Last Text",
                message=message,
                default_text=text,
                ok="Copy",
                cancel="Close",
                dimensions=(480, 220),
            ).run()
        )
        if response.clicked:
            self._presenter.copy_again()

    def _copy_again_clicked(self, _item: rumps.MenuItem) -> None:
        self._presenter.copy_again()

    def _setting_clicked(self, name: str) -> Callable[[rumps.MenuItem], None]:
        def clicked(_item: rumps.MenuItem) -> None:
            current = getattr(self._presenter.settings, name)
            response = self._activate_and_run(
                lambda: rumps.Window(
                    title=SETTING_LABELS[name],
                    message=SETTING_HINTS[name],
                    default_text="" if current is None else str(current),
                    ok="Save",
                    cancel="Cancel",
                    dimensions=(360, 24),
                ).run()
            )
            if response.clicked:
                self._presenter.update_setting(name, response.text)

        return clicked

    def _auto_paste_clicked(self, _item: rumps.MenuItem) -> None:
        self._presenter.update_setting("auto_paste", not self._presenter.settings.auto_paste)

    def _quit_clicked(self, _item: rumps.MenuItem) -> None:
        # terminate_() never returns, so everything that must happen goes first.
        self._on_quit()
        rumps.quit_application()


def _force_activate() -> None:
    """Every documented lever for making this process frontmost, best-effort.

    `NSApplication.activateIgnoringOtherApps_` is the normal AppKit call. The
    Carbon `SetFrontProcessWithOptions` fallback is a workaround some developers
    reported working where the AppKit call didn't (Apple Developer Forums
    thread 739524); it is deprecated and the symbols may be absent on a future
    macOS, so every step is best-effort and never raises. See `_activate_and_run`
    for why neither is guaranteed to work on current macOS.
    """
    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    try:
        import ctypes
        import ctypes.util

        path = ctypes.util.find_library("ApplicationServices")
        if not path:
            return
        lib = ctypes.CDLL(path)

        class _PSN(ctypes.Structure):
            _fields_ = (("high", ctypes.c_uint32), ("low", ctypes.c_uint32))

        psn = _PSN()
        if lib.GetCurrentProcess(ctypes.byref(psn)) == 0:
            kSetFrontProcessFrontWindowOnly = 1
            lib.SetFrontProcessWithOptions(ctypes.byref(psn), kSetFrontProcessFrontWindowOnly)
    except (OSError, AttributeError):
        logger.debug("Carbon activation fallback unavailable", exc_info=True)


def _enable(item: rumps.MenuItem, callback: Callable[..., None], enabled: bool) -> None:
    """rumps greys out an item without a callback; there is no separate enabled flag."""
    item.set_callback(callback if enabled else None)


def _pretty(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def paste() -> None:
    """Press Cmd+V in the frontmost app. Needs the same Accessibility grant as the hotkey."""
    time.sleep(0.05)  # let the pasteboard settle before the target app reads it
    controller = keyboard.Controller()
    with controller.pressed(keyboard.Key.cmd):
        controller.press("v")
        controller.release("v")


def load_engine(settings: Settings, presenter: MenuBarPresenter) -> None:
    """Build the AI processor, recorder and engine; report to the presenter either way."""
    try:
        ai = AIProcessor(settings)
    except Exception as e:
        logger.exception("Could not load Whisper model")
        presenter.fail(f"Could not load Whisper model: {e}")
        return
    warning = ai.check_ollama()
    logger.info("Engine ready (hotkey %s)%s", settings.hotkey, f"; {warning}" if warning else "")
    presenter.attach(Engine(AudioRecorder(), ai), warning)


class HotkeyBinding:
    """Owns the pynput listener so the hotkey can be re-registered from Settings."""

    def __init__(self, on_activate: Callable[[], None]) -> None:
        self._on_activate = on_activate
        self._listener: keyboard.GlobalHotKeys | None = None

    def bind(self, hotkey: str) -> None:
        """Replace the current binding. Raises `ValueError` (from pynput) if `hotkey` is invalid."""
        keyboard.HotKey.parse(hotkey)  # validate before touching the running listener
        listener = keyboard.GlobalHotKeys({hotkey: self._on_activate})
        listener.daemon = True
        self.stop()
        listener.start()
        listener.wait()  # stopping a listener whose run loop is not up yet aborts the process
        self._listener = listener

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `localwhisper-gui` console script."""
    try:
        settings = load_settings(argv)
    except ConfigError as e:
        # No UI exists yet; stderr is the only channel for a startup error.
        sys.stderr.write(f"Configuration error: {e}\n")
        return 2
    config_path = resolve_config_path(build_parser().parse_args(argv))

    if settings.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
        )

    hotkey: HotkeyBinding | None = None
    app: MenuBarApp | None = None
    presenter = MenuBarPresenter(
        settings,
        paste=paste,
        notify=notify,
        alert=lambda title, message: app.show_alert(title, message) if app else None,
        save=lambda name, value: save_setting(name, value, config_path),
        rebind_hotkey=lambda combo: hotkey.bind(combo) if hotkey else None,
    )
    hotkey = HotkeyBinding(presenter.toggle_record)
    hotkey.bind(settings.hotkey)

    threading.Thread(
        target=load_engine, args=(settings, presenter), name="localwhisper-loader", daemon=True
    ).start()

    closed = False

    def close() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        hotkey.stop()
        if not presenter.close():
            logger.warning("A job was still running; exiting without waiting for it.")

    app = MenuBarApp(presenter, on_quit=close, config_path=config_path)
    # rumps activates once, early, before the status item/delegate exist; try again now
    # that setup is complete and the run loop is about to start (best-effort, see
    # `_force_activate`'s docstring for why this is not a guaranteed fix).
    rumps.events.before_start.register(_force_activate)
    try:
        app.run()
    finally:
        close()  # Ctrl+C from a terminal stops the run loop instead of going through Quit
    return 0


if __name__ == "__main__":
    sys.exit(main())
