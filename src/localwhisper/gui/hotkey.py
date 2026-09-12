"""Native global hotkey for the menu bar app: Carbon `RegisterEventHotKey` via ctypes.

Why not pynput (which the CLI keeps): pynput *observes* key events through a
Quartz event tap, so the keystroke still reaches the frontmost app (review F-18:
Cmd+Shift+G also opens "Go to Folder" in Finder) and the process needs both
Accessibility and Input Monitoring. A registered hot key is *consumed* by the
system on this app's behalf and needs no privacy permission at all.

`RegisterEventHotKey` is Carbon, but it is the API every current macOS hotkey
library still uses (there is no AppKit replacement), and PyObjC ships no Carbon
bindings, hence ctypes. The handler runs on the main thread, dispatched by the
NSApplication run loop that `rumps` owns; nothing here works without that loop.

Key codes are virtual key *positions* on an ANSI (US) keyboard: on layouts that
move letters (German Z/Y, AZERTY) the combination follows the physical key, not
the printed character.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Carbon modifier masks (Events.h).
CMD_KEY = 0x0100
SHIFT_KEY = 0x0200
OPTION_KEY = 0x0800
CONTROL_KEY = 0x1000

_MODIFIERS: dict[str, int] = {
    "cmd": CMD_KEY,
    "shift": SHIFT_KEY,
    "alt": OPTION_KEY,
    "option": OPTION_KEY,
    "ctrl": CONTROL_KEY,
}

# kVK_* virtual key codes (HIToolbox Events.h), ANSI layout.
# fmt: off
_KEY_CODES: dict[str, int] = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
    "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
    "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18,
    "9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20,
    "[": 0x21, "i": 0x22, "p": 0x23, "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29,
    "\\": 0x2A, ",": 0x2B, "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "`": 0x32, "space": 0x31,
    "tab": 0x30, "enter": 0x24, "esc": 0x35, "backspace": 0x33, "delete": 0x75, "home": 0x73,
    "end": 0x77, "page_up": 0x74, "page_down": 0x79, "left": 0x7B, "right": 0x7C, "down": 0x7D,
    "up": 0x7E, "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61,
    "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F, "f13": 0x69,
    "f14": 0x6B, "f15": 0x71, "f16": 0x6A, "f17": 0x40, "f18": 0x4F, "f19": 0x50, "f20": 0x5A,
}
# fmt: on


def parse_hotkey(hotkey: str) -> tuple[int, int]:
    """Translate pynput syntax (`<cmd>+<shift>+g`) to `(virtual key code, Carbon modifiers)`.

    Raises:
        ValueError: Unknown key or modifier, no modifier, or not exactly one key.
    """
    modifiers = 0
    keys: list[int] = []
    for raw in hotkey.lower().split("+"):
        token = raw.strip()
        name = token[1:-1] if token.startswith("<") and token.endswith(">") else token
        # pynput spells side-specific modifiers `<cmd_l>`, `<alt_r>`; a hot key cannot tell.
        base = name.removesuffix("_l").removesuffix("_r").removesuffix("_gr")
        if token.startswith("<") and base in _MODIFIERS:
            modifiers |= _MODIFIERS[base]
        elif (len(name) == 1 or token.startswith("<")) and name in _KEY_CODES:
            keys.append(_KEY_CODES[name])
        else:
            raise ValueError(f"unsupported key {token!r} in {hotkey!r}")
    if len(keys) != 1:
        raise ValueError(f"{hotkey!r} must name exactly one non-modifier key")
    if not modifiers:
        raise ValueError(f"{hotkey!r} needs at least one modifier")
    return keys[0], modifiers


class _EventTypeSpec(ctypes.Structure):
    _fields_ = (("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32))


class _EventHotKeyID(ctypes.Structure):
    _fields_ = (("signature", ctypes.c_uint32), ("id", ctypes.c_uint32))


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def _four_cc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


K_EVENT_CLASS_KEYBOARD = _four_cc("keyb")
K_EVENT_HOT_KEY_PRESSED = 5
SIGNATURE = _four_cc("LWsp")


class NativeHotkey:
    """One registered hot key at a time, re-bindable. Main thread only."""

    def __init__(self, on_activate: Callable[[], None]) -> None:
        """Install the Carbon event handler. Raises `OSError` if Carbon cannot be loaded."""
        path = ctypes.util.find_library("Carbon")
        if not path:
            raise OSError("Carbon framework not found")
        carbon = ctypes.CDLL(path)
        carbon.GetApplicationEventTarget.restype = ctypes.c_void_p
        carbon.GetApplicationEventTarget.argtypes = ()
        carbon.InstallEventHandler.restype = ctypes.c_int32
        carbon.InstallEventHandler.argtypes = (
            ctypes.c_void_p,
            _HANDLER,
            ctypes.c_ulong,
            ctypes.POINTER(_EventTypeSpec),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        carbon.RegisterEventHotKey.restype = ctypes.c_int32
        carbon.RegisterEventHotKey.argtypes = (
            ctypes.c_uint32,
            ctypes.c_uint32,
            _EventHotKeyID,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        )
        carbon.UnregisterEventHotKey.restype = ctypes.c_int32
        carbon.UnregisterEventHotKey.argtypes = (ctypes.c_void_p,)
        self._carbon = carbon
        self._on_activate = on_activate
        self._ref: ctypes.c_void_p | None = None
        self._hotkey: str | None = None

        # Keep a reference: ctypes frees the trampoline when the Python object is collected.
        self._handler = _HANDLER(self._handle)
        spec = _EventTypeSpec(K_EVENT_CLASS_KEYBOARD, K_EVENT_HOT_KEY_PRESSED)
        status = carbon.InstallEventHandler(
            carbon.GetApplicationEventTarget(), self._handler, 1, ctypes.byref(spec), None, None
        )
        if status != 0:
            raise OSError(f"InstallEventHandler failed (OSStatus {status})")

    def _handle(self, _call: int | None, _event: int | None, _user: int | None) -> int:
        try:
            self._on_activate()
        except Exception:  # an exception must not unwind through Carbon's C frames
            logger.exception("Hotkey callback failed")
        return 0  # noErr: handled, do not pass the event on

    def bind(self, hotkey: str) -> None:
        """Register `hotkey`, replacing the current one; the old one stays if this fails.

        Raises:
            ValueError: `hotkey` cannot be expressed as a hot key, or macOS refused it
                (already registered by another app, or a combination macOS reserves,
                such as Option/Shift-only hot keys since macOS 15).
        """
        code, modifiers = parse_hotkey(hotkey)
        previous = self._hotkey
        self.stop()
        try:
            self._register(hotkey, code, modifiers)
        except ValueError:
            if previous is not None:
                self._register(previous, *parse_hotkey(previous))
            raise

    def _register(self, hotkey: str, code: int, modifiers: int) -> None:
        ref = ctypes.c_void_p()
        status = self._carbon.RegisterEventHotKey(
            code,
            modifiers,
            _EventHotKeyID(SIGNATURE, 1),
            self._carbon.GetApplicationEventTarget(),
            0,
            ctypes.byref(ref),
        )
        if status != 0:
            raise ValueError(f"macOS refused hot key {hotkey!r} (OSStatus {status})")
        self._ref = ref
        self._hotkey = hotkey

    def stop(self) -> None:
        if self._ref is not None:
            self._carbon.UnregisterEventHotKey(self._ref)
            self._ref = None
            self._hotkey = None
