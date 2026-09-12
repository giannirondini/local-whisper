"""macOS user notifications for the menu bar app.

Both `UNUserNotificationCenter` and the legacy `NSUserNotification` (what `rumps`
wraps) require the process to run inside an app bundle with a `CFBundleIdentifier`;
from a virtualenv's `python` there is none and both fail. Until the `.app` bundle
exists (UI plan session U4) the notification goes through `osascript`, which
works from any process (the banner is attributed to Script Editor). The text is
passed as script arguments, never interpolated into the script, and only reaches
the local Notification Center.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

_SCRIPT = [
    "-e",
    "on run argv",
    "-e",
    "display notification (item 1 of argv) with title (item 2 of argv)",
    "-e",
    "end run",
]


def notify(title: str, message: str) -> None:
    """Post a notification without blocking; failures are logged, never raised."""
    try:
        subprocess.Popen(
            ["osascript", *_SCRIPT, message, title],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        logger.warning("Could not post notification: %s", e)
