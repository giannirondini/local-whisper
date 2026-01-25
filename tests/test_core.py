import pytest
from unittest.mock import MagicMock, patch
from localwhisper.core import AIProcessor

class TestAIProcessor:
    @pytest.fixture
    def mock_whisper(self):
        with patch('localwhisper.core.WhisperModel') as mock:
            yield mock

    @pytest.fixture
    def processor(self, mock_whisper):
        # Prevent actual model loading during tests
        return AIProcessor(whisper_size="tiny", ollama_model="test-model")

    def test_init_loads_whisper(self, mock_whisper):
        """Test that WhisperModel is initialized with correct parameters."""
        AIProcessor(whisper_size="tiny")
        mock_whisper.assert_called_with("tiny", device="cpu", compute_type="int8")

    def test_transcribe_success(self, processor):
        """Test successful transcription."""
        # Mock the transcribe method of the model instance
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        processor.model.transcribe.return_value = ([mock_segment], None)

        result = processor.transcribe("fake_audio.wav")
        
        assert result == "Hello world"
        processor.model.transcribe.assert_called_once()

    def test_transcribe_failure(self, processor):
        """Test transcription error handling."""
        processor.model.transcribe.side_effect = Exception("Transcription failed")
        
        result = processor.transcribe("fake_audio.wav")
        assert result is None

    @patch('localwhisper.core.requests.post')
    def test_refine_text_success(self, mock_post, processor):
        """Test successful text refinement via Ollama."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Refined text"}
        mock_post.return_value = mock_response

        result = processor.refine_text("raw text")
        
        assert result == "Refined text"
        mock_post.assert_called_once()
        
    @patch('localwhisper.core.requests.post')
    def test_refine_text_with_instruction(self, mock_post, processor):
        """Test refinement with extra instruction."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Translated text"}
        mock_post.return_value = mock_response

        processor.refine_text("raw text", instruction="Translate to Spanish")
        
        # Check if instruction is in the system prompt payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert "Translate to Spanish" in payload['system']
