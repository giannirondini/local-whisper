"""LocalWhisper package."""

import os

__version__ = "1.0.0"

# Third-party telemetry off before any dependency initializes. This must run before
# `faster_whisper` imports onnxruntime (the Silero VAD), so it lives at the package root,
# which every entry point (CLI, menu bar app, PyInstaller bundle) imports first.
#
# onnxruntime >= 1.2x ships Microsoft's 1DS SDK on macOS: it persists a device id under
# ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime, queues events
# (OS, CPU model, model file name and hashes, timings, errors) and uploads them to
# mobile.events.data.microsoft.com. The runtime `disable_telemetry_events()` call is not
# enough: per onnxruntime's posix/telemetry.cc it "leaves the uploader live, so ProcessInfo
# still fires". Only the environment variable, read when ORT initializes, suppresses it.
# Forced, not setdefault: a stray ORT_DISABLE_TELEMETRY=0 must not re-enable it here.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
