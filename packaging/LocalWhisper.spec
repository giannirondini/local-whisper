# PyInstaller spec for LocalWhisper.app. Build with `packaging/build_app.sh`.
# -*- mode: python -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata

from localwhisper import __version__

BUNDLE_ID = "io.github.gianni.localwhisper"

datas, binaries, hiddenimports = [], [], ["rumps"]
# faster_whisper ships the Silero VAD model as package data (loaded by vad_filter=True);
# ctranslate2 / onnxruntime / av carry dylibs PyInstaller's import analysis misses.
for package in ("faster_whisper", "ctranslate2", "onnxruntime", "av", "tokenizers"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
# huggingface_hub and tqdm read their own version through importlib.metadata.
for dist in ("huggingface_hub", "tqdm", "tokenizers", "faster-whisper"):
    datas += copy_metadata(dist)

a = Analysis(
    ["localwhisper_app.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "mypy", "ruff"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalWhisper",
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,  # signed as a whole by build_app.sh
)
coll = COLLECT(exe, a.binaries, a.datas, name="LocalWhisper")

app = BUNDLE(
    coll,
    name="LocalWhisper.app",
    bundle_identifier=BUNDLE_ID,
    version=__version__,
    info_plist={
        "CFBundleName": "LocalWhisper",
        "CFBundleDisplayName": "LocalWhisper",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "13.0",
        # Menu bar agent: no Dock icon, no app menu.
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": (
            "LocalWhisper records your voice to transcribe it on this Mac. "
            "Audio stays in memory and never leaves the machine."
        ),
    },
)
