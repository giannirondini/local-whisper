"""AI processing module for LocalWhisper.

This module handles speech-to-text transcription using Whisper
and text refinement using a local Ollama LLM.
"""

from __future__ import annotations

import fnmatch
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import requests
from faster_whisper import WhisperModel
from faster_whisper.utils import available_models, download_model
from huggingface_hub import HfApi, constants
from huggingface_hub.errors import LocalEntryNotFoundError

from . import prompts
from .audio import SAMPLE_RATE
from .config import Settings

logger = logging.getLogger(__name__)

# Connect timeout for Ollama requests; the read timeout is `Settings.ollama_timeout`
# because the first request after a cold model load can legitimately take a while.
OLLAMA_CONNECT_TIMEOUT: float = 5.0

# Timeout for the startup `/api/tags` probe; it must never delay startup noticeably.
OLLAMA_CHECK_TIMEOUT: float = 3.0

# Recordings shorter than this are treated as accidental presses and never sent to
# Whisper, which tends to hallucinate on near-empty input.
MIN_AUDIO_SECONDS: float = 0.5

# Voice-activity detection drops silent stretches before decoding, which is the main
# defence against "Thank you for watching."-style hallucinations on silence.
VAD_PARAMETERS: dict[str, int] = {"min_silence_duration_ms": 500}


# Files faster-whisper fetches for a model (mirrors `faster_whisper.utils.download_model`).
MODEL_FILE_PATTERNS: tuple[str, ...] = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)

# How often the download progress callback samples the bytes on disk.
DOWNLOAD_POLL_SECONDS: float = 0.5

ProgressCallback = Callable[[int, "int | None"], None]
"""`(downloaded_bytes, total_bytes)`; total is None when the Hub did not report sizes."""


class ModelDownloadError(Exception):
    """Raised when the Whisper model is not cached and cannot be downloaded."""


class TranscriptionError(Exception):
    """Raised when Whisper fails to transcribe an audio file."""


class RefinementError(Exception):
    """Raised when the Ollama refinement request fails or returns an unusable response."""


class AIProcessor:
    """Handles AI-powered transcription and text refinement."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the AI processor. Loads the Whisper model synchronously.

        Args:
            settings: Model names, language, Ollama endpoint and timeouts. Defaults
                to `Settings()`.

        Raises:
            Whatever `faster_whisper.WhisperModel` raises when the model cannot be
            loaded (missing download, unsupported compute type...).
        """
        self.settings: Settings = settings if settings is not None else Settings()

        name = self.settings.whisper_model
        # A cached model is loaded by path: passing the name makes faster-whisper ask the
        # Hugging Face Hub for the latest revision on every launch, even fully offline.
        model = cached_whisper_model(name) or name
        logger.info("Loading Whisper model %s", name)
        # On Apple Silicon, faster-whisper runs on CPU by default.
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        logger.info("Whisper model loaded")

    @property
    def ollama_model(self) -> str:
        return self.settings.ollama_model

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe an in-memory audio buffer to text using Whisper.

        Args:
            audio: Mono float32 samples in [-1, 1] at SAMPLE_RATE (see `audio.py`).

        Returns:
            Transcribed text. An empty string means no speech was detected or the clip
            was too short to be meaningful.

        Raises:
            TranscriptionError: If the Whisper model fails.
        """
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_SECONDS:
            logger.info("Clip too short (%.2fs), skipping transcription", duration)
            return ""

        logger.info("Transcribing %.1fs of audio", duration)
        start = time.time()

        try:
            # beam_size=5 is standard for accuracy.
            # condition_on_previous_text=False stops a bad segment from poisoning the next.
            segments, _info = self.model.transcribe(
                audio,
                beam_size=5,
                language=self.settings.language,
                vad_filter=True,
                vad_parameters=VAD_PARAMETERS,
                condition_on_previous_text=False,
            )
            # Whisper segment texts carry a leading space; strip each before joining.
            transcribed_text = " ".join(segment.text.strip() for segment in segments)
        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}") from e

        logger.info("Transcription complete in %.2fs", time.time() - start)
        return transcribed_text.strip()

    def refine_text(self, text: str, instruction: str | None = None) -> str:
        """Clean up a transcript, or apply an instruction to an already clean text.

        Args:
            text: The text to work on. Without `instruction` this is a raw transcript to
                punctuate and de-filler; with one it is the current text to edit.
            instruction: Optional instruction for how to modify the text.

        Returns:
            The resulting text, never empty.

        Raises:
            RefinementError: If Ollama is unreachable, returns an error, times out, or
                returns an empty or unparsable response.
        """
        logger.info("Sending to Ollama (%s)", self.ollama_model)

        if instruction:
            system_prompt = prompts.EDIT_SYSTEM_PROMPT
            prompt = prompts.build_edit_prompt(text, instruction)
        else:
            system_prompt = prompts.REFINE_SYSTEM_PROMPT
            prompt = prompts.build_refine_prompt(text)

        url = f"{self.settings.ollama_url}/api/generate"
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            # Thinking models spend the whole output budget on hidden reasoning and
            # return "" otherwise. See docs/OLLAMA_MODEL_DECISION.md.
            "think": False,
            "keep_alive": self.settings.keep_alive,
            "options": dict(prompts.OLLAMA_OPTIONS),
        }
        timeout = (OLLAMA_CONNECT_TIMEOUT, self.settings.ollama_timeout)

        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Ollama returns a JSON body like {"error": "model 'x' not found"} on 4xx/5xx.
            detail = _ollama_error_detail(e.response)
            raise RefinementError(f"Ollama returned an error: {detail or e}") from e
        except requests.exceptions.Timeout as e:
            raise RefinementError(
                f"Ollama timed out after {self.settings.ollama_timeout:.0f}s"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RefinementError(f"Cannot reach Ollama at {url}: {e}") from e

        try:
            result = response.json()["response"]
        except (KeyError, ValueError, AttributeError) as e:
            raise RefinementError(f"Unexpected Ollama response: {e}") from e
        if not isinstance(result, str) or not result.strip():
            raise RefinementError(f"Ollama returned an empty response from {self.ollama_model}")
        return result.strip()

    def check_ollama(self) -> str | None:
        """Probe Ollama once at startup. Returns a warning to show the user, or None.

        Never raises and never blocks for more than `OLLAMA_CHECK_TIMEOUT` seconds; a
        failed probe only means refinement will fall back to the raw transcript later.
        """
        url = f"{self.settings.ollama_url}/api/tags"
        try:
            response = requests.get(url, timeout=OLLAMA_CHECK_TIMEOUT)
            response.raise_for_status()
            names = [model["name"] for model in response.json()["models"]]
        except requests.exceptions.RequestException as e:
            return (
                f"Ollama is not reachable at {self.settings.ollama_url} "
                f"({type(e).__name__}). Notes will be copied unrefined until it is running."
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Unexpected /api/tags payload from Ollama; skipping model check")
            return None

        if not _model_available(self.ollama_model, names):
            available = ", ".join(sorted(names)) or "none"
            return (
                f"Model {self.ollama_model!r} is not installed in Ollama (available: {available}). "
                f"Run: ollama pull {self.ollama_model}"
            )
        return None


def _repo_id(name: str) -> str | None:
    """Hub repository for a faster-whisper model name or `org/repo` id; None if unknown."""
    if "/" in name:
        return name
    if name in available_models():
        # faster-whisper's own name → repo mapping; the utils module exposes no public getter.
        from faster_whisper.utils import _MODELS

        return _MODELS[name]
    return None


def cached_whisper_model(name: str) -> str | None:
    """Local directory of `name` if it needs no download (a path, or already cached). No network."""
    if Path(name).expanduser().is_dir():
        return str(Path(name).expanduser())
    if _repo_id(name) is None:
        return None  # let WhisperModel raise its own "invalid model size" error
    try:
        return str(download_model(name, local_files_only=True))
    except (LocalEntryNotFoundError, OSError):
        return None


def download_whisper_model(name: str, on_progress: ProgressCallback | None = None) -> str:
    """Download `name` into the Hugging Face cache and return its local directory.

    This is the one network call LocalWhisper makes besides localhost Ollama. Progress
    is measured from the bytes on disk (faster-whisper disables huggingface_hub's own
    progress bars), sampled every `DOWNLOAD_POLL_SECONDS` on a helper thread.

    Raises:
        ModelDownloadError: Unknown model name, no network, or the Hub refused the request.
    """
    repo_id = _repo_id(name)
    if repo_id is None:
        raise ModelDownloadError(
            f"Unknown Whisper model {name!r}; expected one of: {', '.join(available_models())}"
        )
    total = _download_size(repo_id)
    repo_dir = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"

    done = threading.Event()

    def report() -> None:
        while not done.wait(DOWNLOAD_POLL_SECONDS):
            if on_progress is not None:
                on_progress(_dir_bytes(repo_dir / "blobs"), total)

    reporter = threading.Thread(target=report, name="localwhisper-download", daemon=True)
    reporter.start()
    logger.info("Downloading Whisper model %s from %s", name, repo_id)
    try:
        path = str(download_model(name))
    # Deliberately broad: huggingface_hub's failures are not one hierarchy (HfHubHTTPError,
    # raw httpx transport errors since 1.0 — an offline first launch raises httpx.ConnectError —
    # OSError from the cache), and any of them means the same thing here: no model.
    except Exception as e:
        raise ModelDownloadError(f"Could not download Whisper model {name!r}: {e}") from e
    finally:
        done.set()
        reporter.join()
    if on_progress is not None:
        on_progress(total or _dir_bytes(repo_dir / "blobs"), total)
    logger.info("Whisper model %s downloaded to %s", name, path)
    return path


def _download_size(repo_id: str) -> int | None:
    """Total bytes of the files `download_model` will fetch, or None if the Hub won't say."""
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception:  # noqa: BLE001 - same reasoning as download_whisper_model; size is optional
        logger.info("Could not read the size of %s; progress will have no total", repo_id)
        return None
    sizes = [
        sibling.size or 0
        for sibling in info.siblings or []
        if any(fnmatch.fnmatch(sibling.rfilename, p) for p in MODEL_FILE_PATTERNS)
    ]
    return sum(sizes) or None


def _dir_bytes(path: Path) -> int:
    """Bytes under `path`, including partial `.incomplete` downloads; 0 if it does not exist."""
    try:
        return sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    except OSError:
        return 0


def _model_available(wanted: str, names: list[str]) -> bool:
    """`mistral` matches `mistral:latest`; a tagged name must match exactly."""
    if wanted in names:
        return True
    return ":" not in wanted and f"{wanted}:latest" in names


def _ollama_error_detail(response: requests.Response | None) -> str | None:
    """Extract Ollama's error message from an HTTP error response, if present."""
    if response is None:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return None
