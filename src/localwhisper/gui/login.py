"""Launch at login for the packaged app, through `SMAppService` (macOS 13+).

Only meaningful inside `LocalWhisper.app`: `SMAppService.mainAppService()` registers
the bundle that is running, so from `uv run localwhisper-gui` it reports "not
found". The registration points at the bundle's current location; move the app to
/Applications before turning this on, or the login item breaks when it moves.

The ServiceManagement framework is loaded dynamically with `objc.loadBundle`
instead of adding `pyobjc-framework-ServiceManagement` for three calls.
"""

from __future__ import annotations

import logging
from typing import Any

from . import is_bundled

logger = logging.getLogger(__name__)

# SMAppServiceStatus
NOT_REGISTERED = 0
ENABLED = 1
REQUIRES_APPROVAL = 2
NOT_FOUND = 3
ON_STATES = frozenset({ENABLED, REQUIRES_APPROVAL})
"""Registered: shown as checked even while macOS waits for the user's approval."""

_FRAMEWORK = "/System/Library/Frameworks/ServiceManagement.framework"


class LoginItemError(Exception):
    """Registering or unregistering the login item failed."""


def _service() -> Any:
    import objc  # the `gui` extra; imported lazily so this module is importable without it

    objc.loadBundle("ServiceManagement", {}, bundle_path=_FRAMEWORK)
    cls = objc.lookUpClass("SMAppService")
    # Without the framework wrapper PyObjC does not know the NSError** is an out-param.
    for selector in (b"registerAndReturnError:", b"unregisterAndReturnError:"):
        objc.registerMetaDataForSelector(
            b"SMAppService", selector, {"arguments": {2: {"type_modifier": b"o"}}}
        )
    return cls.mainAppService()


def available() -> bool:
    """Whether the menu should offer the toggle at all: bundled, and the API loads."""
    if not is_bundled():
        return False
    try:
        import objc
    except ImportError:
        return False
    try:
        _service()
    except (ImportError, objc.error) as e:
        logger.warning("Launch at login unavailable: %s", e)
        return False
    return True


def status() -> int:
    return int(_service().status())


def set_enabled(enabled: bool) -> int:
    """Register or unregister; returns the resulting status.

    `REQUIRES_APPROVAL` is a success from the app's side: macOS wants the user to
    allow it under System Settings → General → Login Items.

    Raises:
        LoginItemError: With the system's reason.
    """
    service = _service()
    call = service.registerAndReturnError_ if enabled else service.unregisterAndReturnError_
    ok, error = call(None)
    if not ok:
        reason = error.localizedDescription() if error is not None else "unknown error"
        action = "enable" if enabled else "disable"
        raise LoginItemError(f"Could not {action} launch at login: {reason}")
    return int(service.status())
