from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from localwhisper import prompts
from localwhisper.audio import SAMPLE_RATE
from localwhisper.config import Settings
from localwhisper.core import (
    MIN_AUDIO_SECONDS,
    OLLAMA_CHECK_TIMEOUT,
    OLLAMA_CONNECT_TIMEOUT,
    VAD_PARAMETERS,
    AIProcessor,
    RefinementError,
    TranscriptionError,
)

SETTINGS = Settings(whisper_model="tiny", ollama_model="test-model", ollama_timeout=42.0)

# One second of silence: long enough to pass the minimum-duration gate.
AUDIO = np.zeros(SAMPLE_RATE, dtype=np.float32)


def _segment(text: str) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    return seg


def _ok_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestAIProcessor:
    @pytest.fixture
    def mock_whisper(self):
        with patch("localwhisper.core.WhisperModel") as mock:
            yield mock

    @pytest.fixture
    def processor(self, mock_whisper):
        # Prevent actual model loading during tests
        return AIProcessor(SETTINGS)

    def test_init_loads_whisper(self, mock_whisper):
        """Test that WhisperModel is initialized with correct parameters."""
        AIProcessor(SETTINGS)
        mock_whisper.assert_called_with("tiny", device="cpu", compute_type="int8")

    def test_init_defaults_to_default_settings(self, mock_whisper):
        processor = AIProcessor()
        assert processor.settings == Settings()
        mock_whisper.assert_called_with(Settings().whisper_model, device="cpu", compute_type="int8")

    def test_transcribe_passes_language(self, mock_whisper):
        """F-12: the configured language reaches Whisper (None means auto-detect)."""
        processor = AIProcessor(Settings(whisper_model="base", language="it"))
        processor.model.transcribe.return_value = ([], None)

        processor.transcribe(AUDIO)

        assert processor.model.transcribe.call_args.kwargs["language"] == "it"

    # --- transcribe ---------------------------------------------------------

    def test_transcribe_success(self, processor):
        """Test successful transcription."""
        processor.model.transcribe.return_value = ([_segment("Hello world")], None)

        result = processor.transcribe(AUDIO)

        assert result == "Hello world"
        processor.model.transcribe.assert_called_once()

    def test_transcribe_passes_buffer_with_vad(self, processor):
        """Whisper gets the in-memory buffer, VAD on, no conditioning on previous text (F-14)."""
        processor.model.transcribe.return_value = ([], None)

        processor.transcribe(AUDIO)

        args, kwargs = processor.model.transcribe.call_args
        assert args[0] is AUDIO
        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == VAD_PARAMETERS
        assert kwargs["condition_on_previous_text"] is False

    def test_transcribe_skips_clips_shorter_than_minimum(self, processor):
        """Accidental presses never reach the model (F-14)."""
        too_short = np.zeros(int(SAMPLE_RATE * MIN_AUDIO_SECONDS) - 1, dtype=np.float32)

        assert processor.transcribe(too_short) == ""
        processor.model.transcribe.assert_not_called()

    def test_transcribe_joins_segments_with_single_spaces(self, processor):
        """Whisper segments carry a leading space; joining must not double it (F-21)."""
        processor.model.transcribe.return_value = (
            [_segment(" Hello world."), _segment(" Second sentence.")],
            None,
        )

        assert processor.transcribe(AUDIO) == "Hello world. Second sentence."

    def test_transcribe_empty_returns_empty_string(self, processor):
        """No segments means no speech: an empty string, not an error (F-16)."""
        processor.model.transcribe.return_value = ([], None)

        assert processor.transcribe(AUDIO) == ""

    def test_transcribe_failure_raises(self, processor):
        """Whisper errors are surfaced as TranscriptionError, not as 'no speech' (F-16)."""
        processor.model.transcribe.side_effect = RuntimeError("Transcription failed")

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            processor.transcribe(AUDIO)

    # --- refine_text --------------------------------------------------------

    @patch("localwhisper.core.requests.post")
    def test_refine_text_success(self, mock_post, processor):
        """Test successful text refinement via Ollama."""
        mock_post.return_value = _ok_response({"response": "  Refined text \n"})

        result = processor.refine_text("raw text")

        assert result == "Refined text"
        mock_post.assert_called_once()

    @patch("localwhisper.core.requests.post")
    def test_refine_text_sends_timeout(self, mock_post, processor):
        """The Ollama call must never block forever (F-03); read timeout is configurable (F-12)."""
        mock_post.return_value = _ok_response({"response": "x"})

        processor.refine_text("raw text")

        assert mock_post.call_args.kwargs["timeout"] == (OLLAMA_CONNECT_TIMEOUT, 42.0)

    @patch("localwhisper.core.requests.post")
    def test_refine_text_payload(self, mock_post, processor):
        """F-07/F-20: delimited transcript, think off, keep_alive and sampling options set."""
        mock_post.return_value = _ok_response({"response": "x"})

        processor.refine_text("what should i bring")

        url = mock_post.call_args.args[0]
        payload = mock_post.call_args.kwargs["json"]
        assert url == "http://localhost:11434/api/generate"
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["keep_alive"] == SETTINGS.keep_alive
        assert payload["options"] == prompts.OLLAMA_OPTIONS
        assert payload["system"] == prompts.REFINE_SYSTEM_PROMPT
        assert payload["prompt"] == "Transcript:\n<<<\nwhat should i bring\n>>>"
        assert "Never answer" in payload["system"]
        assert "fillers" in payload["system"]

    @patch("localwhisper.core.requests.post")
    def test_refine_text_uses_configured_url(self, mock_post, mock_whisper):
        mock_post.return_value = _ok_response({"response": "x"})
        processor = AIProcessor(Settings(ollama_url="http://gpu-box:11434/"))

        processor.refine_text("raw text")

        assert mock_post.call_args.args[0] == "http://gpu-box:11434/api/generate"

    @patch("localwhisper.core.requests.post")
    def test_refine_text_with_instruction(self, mock_post, processor):
        """An instruction switches to the edit prompt and wraps the current text."""
        mock_post.return_value = _ok_response({"response": "Translated text"})

        processor.refine_text("Clean text.", instruction="Translate to Spanish")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["system"] == prompts.EDIT_SYSTEM_PROMPT
        assert (
            payload["prompt"] == "Instruction: Translate to Spanish\n\nText:\n<<<\nClean text.\n>>>"
        )

    @patch("localwhisper.core.requests.post")
    def test_refine_text_empty_response_raises(self, mock_post, processor):
        """A thinking model that spends its budget on reasoning returns ""; never store it."""
        mock_post.return_value = _ok_response({"response": "  \n"})

        with pytest.raises(RefinementError, match="empty response"):
            processor.refine_text("raw text")

    @patch("localwhisper.core.requests.post")
    def test_refine_text_connection_error_raises(self, mock_post, processor):
        """Ollama down must raise, never return an error string (F-02)."""
        mock_post.side_effect = requests.exceptions.ConnectionError("refused")

        with pytest.raises(RefinementError, match="Cannot reach Ollama"):
            processor.refine_text("raw text")

    @patch("localwhisper.core.requests.post")
    def test_refine_text_timeout_raises(self, mock_post, processor):
        mock_post.side_effect = requests.exceptions.ReadTimeout("slow")

        with pytest.raises(RefinementError, match="timed out"):
            processor.refine_text("raw text")

    @patch("localwhisper.core.requests.post")
    def test_refine_text_http_error_includes_ollama_detail(self, mock_post, processor):
        """A 404 for an unknown model should surface Ollama's own message."""
        resp = MagicMock()
        resp.json.return_value = {"error": "model 'test-model' not found"}
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error", response=resp
        )
        mock_post.return_value = resp

        with pytest.raises(RefinementError, match="model 'test-model' not found"):
            processor.refine_text("raw text")

    @patch("localwhisper.core.requests.post")
    def test_refine_text_malformed_response_raises(self, mock_post, processor):
        mock_post.return_value = _ok_response({"unexpected": "shape"})

        with pytest.raises(RefinementError, match="Unexpected Ollama response"):
            processor.refine_text("raw text")

    # --- check_ollama -------------------------------------------------------

    @patch("localwhisper.core.requests.get")
    def test_check_ollama_ok_when_model_present(self, mock_get, processor):
        mock_get.return_value = _ok_response({"models": [{"name": "test-model:latest"}]})

        assert processor.check_ollama() is None
        assert mock_get.call_args.args[0] == "http://localhost:11434/api/tags"
        assert mock_get.call_args.kwargs["timeout"] == OLLAMA_CHECK_TIMEOUT

    @patch("localwhisper.core.requests.get")
    def test_check_ollama_tagged_name_must_match_exactly(self, mock_get, mock_whisper):
        processor = AIProcessor(Settings(ollama_model="gemma4:e2b-mlx"))
        mock_get.return_value = _ok_response({"models": [{"name": "gemma4:12b-mlx"}]})

        warning = processor.check_ollama()

        assert warning is not None
        assert "ollama pull gemma4:e2b-mlx" in warning
        assert "gemma4:12b-mlx" in warning

    @patch("localwhisper.core.requests.get")
    def test_check_ollama_unreachable_warns_without_raising(self, mock_get, processor):
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")

        warning = processor.check_ollama()

        assert warning is not None
        assert "not reachable" in warning

    @patch("localwhisper.core.requests.get")
    def test_check_ollama_unexpected_payload_is_silent(self, mock_get, processor):
        mock_get.return_value = _ok_response({"weird": []})

        assert processor.check_ollama() is None
