"""
Unit tests for ChatterBox Voice Capture IS_CHANGED behavior
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


@pytest.mark.unit
class TestVoiceCaptureISChanged:
    """Voice Capture records live audio, so it must never serve a cached result."""

    def test_is_changed_implemented(self):
        """Node class implements the IS_CHANGED hook."""
        assert hasattr(ChatterBoxVoiceCapture, "IS_CHANGED")

    def test_is_changed_forces_rerun(self):
        """NaN != NaN: consecutive calls never compare equal, so ComfyUI's
        cache always misses and the node records fresh audio every queue."""
        assert ChatterBoxVoiceCapture.IS_CHANGED() != ChatterBoxVoiceCapture.IS_CHANGED()
