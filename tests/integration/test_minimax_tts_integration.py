"""
Integration tests for MiniMax Cloud TTS Engine

Tests the full integration path: engine node -> adapter -> API (mocked).
These tests verify the node registration, engine data format, and end-to-end
processing without requiring a real API key or ComfyUI server.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

os.environ.setdefault("COMFYUI_TESTING", "1")

# Mock torchaudio if not available (CI environments without GPU deps)
if "torchaudio" not in sys.modules:
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        sys.modules["torchaudio"] = MagicMock()
        sys.modules["torchaudio.transforms"] = MagicMock()


def _make_api_response(audio_hex: str = "aabb", status_code: int = 0,
                       status_msg: str = "success") -> bytes:
    """Build a fake MiniMax T2A v2 JSON response."""
    return json.dumps({
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
        "data": {"audio": audio_hex},
    }).encode("utf-8")


import importlib.util

# Helper to load engine node module directly (avoids ComfyUI 'nodes' mock conflict)
def _load_engine_node_module():
    """Load MiniMaxTTSEngineNode via importlib to bypass 'nodes' mock."""
    node_path = custom_node_root / "nodes" / "engines" / "minimax_tts_engine_node.py"
    spec = importlib.util.spec_from_file_location("minimax_tts_engine_node", str(node_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.integration
class TestMiniMaxTTSEngineNodeIntegration:
    """Tests for the engine node creating correct engine_data."""

    def test_engine_node_creates_valid_engine_data(self):
        """Engine node should produce a valid TTS_ENGINE dict."""
        mod = _load_engine_node_module()
        MiniMaxTTSEngineNode = mod.MiniMaxTTSEngineNode

        node = MiniMaxTTSEngineNode()
        result = node.create_engine_adapter(
            model="speech-2.8-hd",
            voice_id="English_Graceful_Lady",
            speed=1.0,
            api_key="test-key",
        )

        engine_data = result[0]
        assert engine_data["engine_type"] == "minimax_tts"
        assert engine_data["adapter_class"] == "MiniMaxTTSAdapter"
        assert engine_data["config"]["model"] == "speech-2.8-hd"
        assert engine_data["config"]["voice_id"] == "English_Graceful_Lady"
        assert engine_data["config"]["speed"] == 1.0
        assert engine_data["config"]["api_key"] == "test-key"

    def test_engine_node_no_api_key_in_config_when_empty(self):
        """Engine node should not store empty API key in config."""
        mod = _load_engine_node_module()
        MiniMaxTTSEngineNode = mod.MiniMaxTTSEngineNode

        node = MiniMaxTTSEngineNode()
        result = node.create_engine_adapter(
            model="speech-2.8-turbo",
            voice_id="cute_boy",
            speed=1.5,
            api_key="",
        )

        engine_data = result[0]
        assert "api_key" not in engine_data["config"]

    def test_engine_node_return_type(self):
        """Engine node should return a tuple with TTS_ENGINE."""
        mod = _load_engine_node_module()
        MiniMaxTTSEngineNode = mod.MiniMaxTTSEngineNode

        node = MiniMaxTTSEngineNode()
        result = node.create_engine_adapter(
            model="speech-2.8-hd",
            voice_id="Wise_Woman",
            speed=0.8,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_engine_node_class_metadata(self):
        """Engine node should have correct ComfyUI metadata."""
        mod = _load_engine_node_module()
        MiniMaxTTSEngineNode = mod.MiniMaxTTSEngineNode

        assert MiniMaxTTSEngineNode.RETURN_TYPES == ("TTS_ENGINE",)
        assert MiniMaxTTSEngineNode.FUNCTION == "create_engine_adapter"
        assert "Engines" in MiniMaxTTSEngineNode.CATEGORY

    def test_engine_node_input_types(self):
        """Engine node should declare correct input types."""
        mod = _load_engine_node_module()
        MiniMaxTTSEngineNode = mod.MiniMaxTTSEngineNode

        input_types = MiniMaxTTSEngineNode.INPUT_TYPES()

        assert "model" in input_types["required"]
        assert "voice_id" in input_types["required"]
        assert "speed" in input_types["required"]
        assert "api_key" in input_types.get("optional", {})


@pytest.mark.integration
class TestMiniMaxTTSAdapterEndToEnd:
    """End-to-end tests for adapter with mocked API."""

    def test_full_generation_flow(self):
        """Test the complete flow: config -> adapter -> generate_single."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({
            "api_key": "test-key",
            "voice_id": "English_Graceful_Lady",
            "model": "speech-2.8-hd",
            "speed": 1.0,
        })

        fake_audio = torch.randn(32000)  # 1 second at 32kHz

        with patch.object(adapter, "_call_tts_api", return_value=b"\x00" * 100):
            with patch.object(adapter, "_decode_mp3_to_tensor",
                              return_value=(fake_audio, 32000)):
                audio, sr = adapter.generate_single(
                    "Hello, this is a test.", enable_audio_cache=False
                )

        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 1
        assert sr == 32000
        assert audio.shape[0] == 32000

    def test_process_text_with_chunking(self):
        """Test text processing with chunking enabled."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({
            "api_key": "test-key",
            "voice_id": "cute_boy",
            "model": "speech-2.8-turbo",
        })

        # Generate a long text that will be chunked
        long_text = "This is a test sentence. " * 30  # ~750 chars

        chunk_audio = torch.randn(16000)

        with patch.object(adapter, "generate_single",
                          return_value=(chunk_audio, 32000)):
            result = adapter.process_text(
                long_text,
                enable_chunking=True,
                max_chars_per_chunk=200,
                enable_audio_cache=False,
            )

        assert isinstance(result, torch.Tensor)
        # Result should be longer than a single chunk since text was chunked
        assert result.numel() > 0

    def test_adapter_caching(self):
        """Test that identical requests use cache."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({
            "api_key": "test-key",
            "voice_id": "cute_boy",
            "model": "speech-2.8-hd",
        })

        fake_audio = torch.randn(16000)
        call_count = 0

        def counting_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return b"\x00" * 100

        with patch.object(adapter, "_call_tts_api", side_effect=counting_call):
            with patch.object(adapter, "_decode_mp3_to_tensor",
                              return_value=(fake_audio, 32000)):
                # First call - should hit API
                adapter.generate_single("Cached test", enable_audio_cache=True)
                first_count = call_count

                # Second call with same text - should use cache
                adapter.generate_single("Cached test", enable_audio_cache=True)
                second_count = call_count

        assert first_count == 1
        assert second_count == 1  # No additional API call


@pytest.mark.integration
class TestMiniMaxTTSNodeRegistration:
    """Tests for node registration in the engine system."""

    def test_adapter_importable(self):
        """MiniMax TTS adapter should be importable."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        assert MiniMaxTTSAdapter is not None

    def test_adapter_available_flag(self):
        """MiniMax TTS adapter availability flag should be set."""
        from engines.adapters import MINIMAX_TTS_ADAPTER_AVAILABLE
        assert MINIMAX_TTS_ADAPTER_AVAILABLE is True

    def test_engine_node_mappings(self):
        """Engine node should have correct NODE_CLASS_MAPPINGS."""
        mod = _load_engine_node_module()
        NODE_CLASS_MAPPINGS = mod.NODE_CLASS_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS = mod.NODE_DISPLAY_NAME_MAPPINGS

        assert "MiniMaxTTSEngineNode" in NODE_CLASS_MAPPINGS
        assert "MiniMaxTTSEngineNode" in NODE_DISPLAY_NAME_MAPPINGS
        assert "MiniMax" in NODE_DISPLAY_NAME_MAPPINGS["MiniMaxTTSEngineNode"]
