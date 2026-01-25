import pytest
from unittest.mock import patch, MagicMock
from localwhisper.audio import AudioRecorder

class TestAudioRecorder:
    @pytest.fixture
    def mock_pyaudio(self):
        with patch('localwhisper.audio.pyaudio.PyAudio') as mock:
            yield mock

    def test_init(self, mock_pyaudio):
        """Test initialization of AudioRecorder."""
        recorder = AudioRecorder()
        assert recorder.recording is False
        assert recorder.frames == []
        assert recorder.p is not None

    def test_start_recording(self, mock_pyaudio):
        """Test starting the recording stream."""
        recorder = AudioRecorder()
        
        with patch('threading.Thread'):  # prevent thread from actually starting
            recorder.start_recording()
            
        assert recorder.recording is True
        recorder.p.open.assert_called_once()
        
    def test_stop_recording_no_audio(self, mock_pyaudio):
        """Test stopping recording when no frames were captured."""
        recorder = AudioRecorder()
        recorder.recording = True
        mock_stream = MagicMock()
        recorder.stream = mock_stream
        
        result = recorder.stop_recording()
        
        assert result is None
        assert recorder.recording is False
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch('localwhisper.audio.wave.open')
    @patch('localwhisper.audio.tempfile.NamedTemporaryFile')
    def test_stop_recording_saves_file(self, mock_temp_file, mock_wave, mock_pyaudio):
        """Test that stopping recording saves a wav file if frames exist."""
        recorder = AudioRecorder()
        recorder.recording = True
        recorder.stream = MagicMock()
        recorder.frames = [b'some_audio_data']
        
        # Mock temp file context manager
        mock_temp_obj = MagicMock()
        mock_temp_obj.name = "/tmp/test.wav"
        mock_temp_file.return_value.__enter__.return_value = mock_temp_obj
        
        result = recorder.stop_recording()
        
        assert result == "/tmp/test.wav"
        mock_wave.assert_called_once()
