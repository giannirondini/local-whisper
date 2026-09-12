"""Menu bar presenter for LocalWhisper (see docs/UI_PLAN.md).

`presenter.py` is toolkit-free and unit-tested; `menubar.py` is the thin `rumps`
shell around it and the `localwhisper-gui` entry point. `hotkey.py` (native Carbon
hot key) and `login.py` (launch at login) are the macOS integrations the `.app`
bundle built from `packaging/` relies on.
"""

from __future__ import annotations

import sys


def is_bundled() -> bool:
    """True when running from the PyInstaller-built `LocalWhisper.app`."""
    return bool(getattr(sys, "frozen", False))
