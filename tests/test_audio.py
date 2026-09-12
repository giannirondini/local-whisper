from unittest.mock import MagicMock, patch

import numpy as np
import pyaudio
import pytest

from localwhisper.audio import (
    SAMPLE_RATE,
    AudioDeviceError,
    AudioRecorder,
    pcm_to_float32,
    save_wav,
)

# Callback arguments PortAudio would pass; only `status` matters to us.
_TIME_INFO: dict[str, float] = {}


class TestAudioRecorder:
    @pytest.fixture
    def mock_pyaudio(self):
        with patch("localwhisper.audio.pyaudio.PyAudio") as mock:
            yield mock

    def test_init(self, mock_pyaudio):
        """Test initialization of AudioRecorder."""
        recorder = AudioRecorder()
        assert recorder.recording is False
        assert recorder.frames == []
        assert recorder.dropped_chunks == 0
        assert recorder.p is not None

    def test_start_recording_uses_callback_mode(self, mock_pyaudio):
        """The stream is opened with a callback: no Python read loop, no thread (F-04)."""
        recorder = AudioRecorder()

        recorder.start_recording()

        assert recorder.recording is True
        kwargs = recorder.p.open.call_args.kwargs
        assert kwargs["stream_callback"] == recorder._callback
        assert kwargs["input"] is True
        assert kwargs["rate"] == SAMPLE_RATE

    def test_start_recording_resets_previous_session(self, mock_pyaudio):
        recorder = AudioRecorder()
        recorder.frames = [b"old"]
        recorder.dropped_chunks = 3

        recorder.start_recording()

        assert recorder.frames == []
        assert recorder.dropped_chunks == 0

    def test_start_recording_stream_error_raises_and_resets_state(self, mock_pyaudio):
        """PortAudio failures surface as AudioDeviceError and leave the recorder idle (F-09)."""
        recorder = AudioRecorder()
        recorder.p.open.side_effect = OSError("no input device")

        with pytest.raises(AudioDeviceError, match="no input device"):
            recorder.start_recording()

        assert recorder.recording is False
        assert recorder.stream is None

    def test_terminate_closes_stream_and_releases_portaudio(self, mock_pyaudio):
        """F-11: PortAudio is terminated exactly once, even if called twice."""
        recorder = AudioRecorder()
        recorder.start_recording()
        stream = recorder.stream

        recorder.terminate()
        recorder.terminate()

        stream.close.assert_called_once()
        recorder.p.terminate.assert_called_once()
        with pytest.raises(AudioDeviceError):
            recorder.start_recording()

    def test_callback_appends_and_continues(self, mock_pyaudio):
        recorder = AudioRecorder()

        result = recorder._callback(b"\x01\x00", 1, _TIME_INFO, 0)

        assert recorder.frames == [b"\x01\x00"]
        assert result == (None, pyaudio.paContinue)

    def test_callback_counts_overflow_instead_of_failing(self, mock_pyaudio):
        """An input overflow is recorded as a warning counter, never an exception (F-05)."""
        recorder = AudioRecorder()

        recorder._callback(b"\x01\x00", 1, _TIME_INFO, pyaudio.paInputOverflow)

        assert recorder.dropped_chunks == 1
        assert recorder.frames == [b"\x01\x00"]  # data is still kept

    def test_stop_recording_no_audio(self, mock_pyaudio):
        """Test stopping recording when no frames were captured."""
        recorder = AudioRecorder()
        recorder.recording = True
        mock_stream = MagicMock()
        recorder.stream = mock_stream

        result = recorder.stop_recording()

        assert result is None
        assert recorder.recording is False
        assert recorder.stream is None
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_stop_recording_returns_float32_buffer(self, mock_pyaudio):
        """Recorded PCM comes back as normalised float32 samples, no file involved (F-13)."""
        recorder = AudioRecorder()
        recorder.recording = True
        recorder.stream = MagicMock()
        # int16 samples: 0, 16384, -32768
        recorder.frames = [np.array([0, 16384], dtype=np.int16).tobytes(), b"\x00\x80"]

        result = recorder.stop_recording()

        assert result is not None
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [0.0, 0.5, -1.0])
        assert recorder.frames == []  # buffer released

    def test_stop_recording_survives_stream_close_error(self, mock_pyaudio):
        recorder = AudioRecorder()
        recorder.stream = MagicMock()
        recorder.stream.stop_stream.side_effect = OSError("already closed")
        recorder.frames = [b"\x00\x00"]

        result = recorder.stop_recording()

        assert result is not None
        assert recorder.stream is None

    def test_duration_seconds(self, mock_pyaudio):
        recorder = AudioRecorder()
        recorder.frames = [b"\x00\x00" * SAMPLE_RATE]  # one second of int16 silence

        assert recorder.duration_seconds() == pytest.approx(1.0)


class TestHelpers:
    def test_pcm_to_float32_roundtrip_range(self):
        pcm = np.array([-32768, 0, 32767], dtype=np.int16).tobytes()

        out = pcm_to_float32(pcm)

        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [-1.0, 0.0, 32767 / 32768])

    def test_save_wav_writes_16bit_mono(self, tmp_path):
        import wave

        path = tmp_path / "out.wav"
        audio = np.array([0.0, 0.5, -1.0], dtype=np.float32)

        save_wav(audio, str(path))

        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
            frames = np.frombuffer(wf.readframes(3), dtype=np.int16)
        np.testing.assert_array_equal(frames, [0, 16384, -32768])
