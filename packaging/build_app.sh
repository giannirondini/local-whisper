#!/usr/bin/env bash
# Build dist/LocalWhisper.app with PyInstaller and sign it.
#
#   packaging/build_app.sh                 # ad-hoc signature (personal use)
#   CODESIGN_IDENTITY="LocalWhisper Dev" packaging/build_app.sh
#
# An ad-hoc signature changes on every build, so macOS treats each build as a new
# app and asks for Microphone/Accessibility again. A stable identity (even a
# self-signed one from Keychain Access) keeps the grants across rebuilds.
set -euo pipefail

cd "$(dirname "$0")/.."
identity="${CODESIGN_IDENTITY:--}"

uv sync --frozen --all-extras --group package
uv run --frozen --all-extras --group package \
    pyinstaller --noconfirm --clean \
    --distpath dist --workpath build/pyinstaller \
    packaging/LocalWhisper.spec

app="dist/LocalWhisper.app"
# Sign inside-out is what --deep approximates; fine for a non-notarized personal build.
codesign --force --deep --timestamp=none --sign "$identity" "$app"
codesign --verify --deep --strict "$app"
echo "Built $app ($(du -sh "$app" | cut -f1)), signed with identity '$identity'."
