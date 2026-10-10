"""
Unit tests for Voice Capture saved audio and workflow caching
Tests nodes/audio/recorder_node.py without requiring ComfyUI server
"""

import pytest
import sys
import os
import importlib.util
from pathlib import Path

# Add custom node root to path BEFORE any project imports
# This avoids triggering the full node loading chain
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Set up minimal environment to avoid ComfyUI imports
os.environ.setdefault('COMFYUI_TESTING', '1')

# Load the recorder module directly using importlib to bypass package __init__.py
# (sys.modules['nodes'] is a MagicMock from conftest, so package imports fail)
recorder_path = custom_node_root / "nodes" / "audio" / "recorder_node.py"
spec = importlib.util.spec_from_file_location("recorder_node_module", recorder_path)
recorder_node_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder_node_module)

ChatterBoxVoiceCapture = recorder_node_module.ChatterBoxVoiceCapture


@pytest.fixture
def saved_recording(monkeypatch, tmp_path):
    import folder_paths
    import numpy as np
    import soundfile as sf
    from utils.audio.voice_capture import recording_path

    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    recording = "a" * 32
    path = recording_path(recording)
    path.parent.mkdir()
    sf.write(str(path), np.linspace(-0.2, 0.2, 16000, dtype=np.float32), 16000, subtype="FLOAT")
    return recording, path


@pytest.mark.unit
class TestVoiceCaptureSavedRecording:
    def test_unchanged_recording_is_cacheable(self, saved_recording):
        recording, _ = saved_recording
        assert ChatterBoxVoiceCapture.IS_CHANGED(recording=recording) == ChatterBoxVoiceCapture.IS_CHANGED(recording=recording)

    def test_missing_recording_invalidates_cache(self, saved_recording):
        recording, path = saved_recording
        signature = ChatterBoxVoiceCapture.IS_CHANGED(recording=recording)
        path.unlink()
        assert signature != ChatterBoxVoiceCapture.IS_CHANGED(recording=recording)

    def test_execution_uses_saved_audio_without_microphone(self, saved_recording, monkeypatch):
        from utils.audio import voice_capture
        monkeypatch.setattr(voice_capture, "_load_sounddevice", lambda: pytest.fail("Workflow execution must not open the microphone"))
        recording, _ = saved_recording
        audio, = ChatterBoxVoiceCapture().capture_voice_audio(recording=recording, trim_start=0.25, trim_end=0.75)
        assert audio["sample_rate"] == 16000
        assert tuple(audio["waveform"].shape) == (1, 1, 8000)

    def test_running_without_recording_has_clear_error(self):
        with pytest.raises(ValueError, match="Start Recording"):
            ChatterBoxVoiceCapture().capture_voice_audio()

    def test_invalid_trim_is_rejected(self, saved_recording):
        recording, _ = saved_recording
        with pytest.raises(ValueError, match="trim range"):
            ChatterBoxVoiceCapture().capture_voice_audio(recording=recording, trim_start=0.75, trim_end=0.25)

    def test_legacy_widget_order_is_preserved(self):
        inputs = ChatterBoxVoiceCapture.INPUT_TYPES()
        assert list(inputs["required"]) == [
            "voice_device", "voice_sample_rate", "voice_max_recording_time", "voice_volume_gain",
            "voice_silence_threshold", "voice_silence_duration", "voice_auto_normalize",
        ]
        assert list(inputs["optional"])[0] == "voice_trigger"

    def test_recording_id_cannot_escape_input_folder(self):
        from utils.audio.voice_capture import recording_path
        with pytest.raises(ValueError):
            recording_path("../../outside")
