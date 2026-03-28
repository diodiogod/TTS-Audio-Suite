"""
Unit tests for MiniMax Cloud TTS Engine Adapter

Tests engines/adapters/minimax_tts_adapter.py without requiring ComfyUI server
or actual API calls.
"""

import json
import os
import sys
import struct
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Set up minimal environment to avoid ComfyUI imports
os.environ.setdefault("COMFYUI_TESTING", "1")

# Mock torchaudio if not available (CI environments without GPU deps)
if "torchaudio" not in sys.modules:
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        sys.modules["torchaudio"] = MagicMock()
        sys.modules["torchaudio.transforms"] = MagicMock()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_api_response(audio_hex: str = "", status_code: int = 0,
                       status_msg: str = "success") -> bytes:
    """Build a fake MiniMax T2A v2 JSON response."""
    return json.dumps({
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
        "data": {"audio": audio_hex},
    }).encode("utf-8")


def _hex_for_silence_wav(num_samples: int = 1600, sample_rate: int = 32000) -> str:
    """
    Create a minimal WAV file in memory and return its hex representation.
    This avoids needing actual MP3 encoding for unit tests.
    """
    # We'll use the torchaudio mock instead — tests that need real decoding
    # will mock _decode_mp3_to_tensor directly.
    return "00" * 100  # Placeholder hex


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.mark.unit
class TestMiniMaxTTSAdapterConfig:
    """Tests for adapter configuration."""

    def test_init_with_empty_config(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({})
        assert adapter.config == {}

    def test_init_preserves_config(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        cfg = {"model": "speech-2.8-turbo", "voice_id": "cute_boy"}
        adapter = MiniMaxTTSAdapter(cfg)
        assert adapter.config["model"] == "speech-2.8-turbo"
        assert adapter.config["voice_id"] == "cute_boy"

    def test_init_copies_config(self):
        """Adapter should copy config to avoid external mutations."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        cfg = {"model": "speech-2.8-hd"}
        adapter = MiniMaxTTSAdapter(cfg)
        cfg["model"] = "changed"
        assert adapter.config["model"] == "speech-2.8-hd"

    def test_update_config(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({"model": "speech-2.8-hd"})
        adapter.update_config({"model": "speech-2.8-turbo", "speed": 1.5})
        assert adapter.config["model"] == "speech-2.8-turbo"
        assert adapter.config["speed"] == 1.5

    def test_update_config_copies(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({})
        new_cfg = {"model": "speech-2.8-hd"}
        adapter.update_config(new_cfg)
        new_cfg["model"] = "mutated"
        assert adapter.config["model"] == "speech-2.8-hd"


@pytest.mark.unit
class TestMiniMaxTTSAdapterAPIKey:
    """Tests for API key resolution."""

    def test_get_api_key_from_config(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({"api_key": "test-key-123"})
        assert adapter._get_api_key() == "test-key-123"

    def test_get_api_key_from_env(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({})
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key-456"}):
            assert adapter._get_api_key() == "env-key-456"

    def test_get_api_key_config_priority(self):
        """Config API key should take priority over env."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({"api_key": "config-key"})
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            assert adapter._get_api_key() == "config-key"

    def test_get_api_key_missing_raises(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
        adapter = MiniMaxTTSAdapter({})
        with patch.dict(os.environ, {}, clear=True):
            # Remove MINIMAX_API_KEY if present
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(ValueError, match="API key not found"):
                adapter._get_api_key()


@pytest.mark.unit
class TestMiniMaxTTSAdapterAPICall:
    """Tests for API request building and response parsing."""

    def test_call_tts_api_builds_correct_payload(self):
        """Verify the API call sends the right JSON payload."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "test-key"})

        captured_request = {}

        def mock_urlopen(req, timeout=None):
            captured_request["url"] = req.full_url
            captured_request["data"] = json.loads(req.data.decode("utf-8"))
            captured_request["headers"] = dict(req.headers)

            # Return valid response
            resp = MagicMock()
            resp.read.return_value = _make_api_response(audio_hex="aabbcc")
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = adapter._call_tts_api("Hello world", "cute_boy", "speech-2.8-hd", 1.2)

        assert captured_request["url"] == "https://api.minimax.io/v1/t2a_v2"
        payload = captured_request["data"]
        assert payload["model"] == "speech-2.8-hd"
        assert payload["text"] == "Hello world"
        assert payload["voice_setting"]["voice_id"] == "cute_boy"
        assert payload["voice_setting"]["speed"] == 1.2
        assert payload["audio_setting"]["format"] == "mp3"
        assert result == bytes.fromhex("aabbcc")

    def test_call_tts_api_clamps_speed(self):
        """Speed should be clamped to [0.5, 2.0]."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "test-key"})
        captured_payload = {}

        def mock_urlopen(req, timeout=None):
            captured_payload.update(json.loads(req.data.decode("utf-8")))
            resp = MagicMock()
            resp.read.return_value = _make_api_response(audio_hex="aa")
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            adapter._call_tts_api("test", "cute_boy", "speech-2.8-hd", 0.1)
            assert captured_payload["voice_setting"]["speed"] == 0.5

            adapter._call_tts_api("test", "cute_boy", "speech-2.8-hd", 5.0)
            assert captured_payload["voice_setting"]["speed"] == 2.0

    def test_call_tts_api_error_response(self):
        """API error response should raise RuntimeError."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "test-key"})

        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.read.return_value = _make_api_response(
                status_code=1001, status_msg="Invalid API key"
            )
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with pytest.raises(RuntimeError, match="Invalid API key"):
                adapter._call_tts_api("test", "cute_boy", "speech-2.8-hd", 1.0)

    def test_call_tts_api_empty_audio(self):
        """Empty audio hex should raise RuntimeError."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "test-key"})

        def mock_urlopen(req, timeout=None):
            resp = MagicMock()
            resp.read.return_value = _make_api_response(audio_hex="")
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            with pytest.raises(RuntimeError, match="empty audio"):
                adapter._call_tts_api("test", "cute_boy", "speech-2.8-hd", 1.0)

    def test_call_tts_api_network_error(self):
        """Network errors should be wrapped in RuntimeError."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "test-key"})

        with patch("urllib.request.urlopen", side_effect=ConnectionError("timeout")):
            with pytest.raises(RuntimeError, match="request failed"):
                adapter._call_tts_api("test", "cute_boy", "speech-2.8-hd", 1.0)


@pytest.mark.unit
class TestMiniMaxTTSAdapterGenerate:
    """Tests for generate_single method."""

    def test_generate_single_returns_tensor(self):
        """generate_single should return (tensor, sample_rate) tuple."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({
            "api_key": "test-key",
            "voice_id": "cute_boy",
            "model": "speech-2.8-hd",
        })

        fake_audio = torch.randn(16000)

        with patch.object(adapter, "_call_tts_api", return_value=b"\x00" * 100):
            with patch.object(adapter, "_decode_mp3_to_tensor",
                              return_value=(fake_audio, 32000)):
                audio, sr = adapter.generate_single("Hello", enable_audio_cache=False)

        assert isinstance(audio, torch.Tensor)
        assert audio.shape == fake_audio.shape
        assert sr == 32000

    def test_generate_single_uses_config_defaults(self):
        """generate_single should use config values when kwargs not provided."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({
            "api_key": "k",
            "voice_id": "Deep_Voice_Man",
            "model": "speech-2.8-turbo",
            "speed": 1.5,
        })

        captured_args = {}
        original_call = adapter._call_tts_api

        def capture_call(text, voice_id, model, speed):
            captured_args.update({
                "voice_id": voice_id,
                "model": model,
                "speed": speed,
            })
            return b"\x00" * 100

        with patch.object(adapter, "_call_tts_api", side_effect=capture_call):
            with patch.object(adapter, "_decode_mp3_to_tensor",
                              return_value=(torch.randn(100), 32000)):
                adapter.generate_single("test", enable_audio_cache=False)

        assert captured_args["voice_id"] == "Deep_Voice_Man"
        assert captured_args["model"] == "speech-2.8-turbo"
        assert captured_args["speed"] == 1.5

    def test_generate_single_kwargs_override(self):
        """generate_single kwargs should override config."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({
            "api_key": "k",
            "voice_id": "cute_boy",
            "model": "speech-2.8-hd",
        })

        captured_args = {}

        def capture_call(text, voice_id, model, speed):
            captured_args.update({"voice_id": voice_id, "model": model})
            return b"\x00" * 100

        with patch.object(adapter, "_call_tts_api", side_effect=capture_call):
            with patch.object(adapter, "_decode_mp3_to_tensor",
                              return_value=(torch.randn(100), 32000)):
                adapter.generate_single(
                    "test", enable_audio_cache=False,
                    voice_id="Wise_Woman", model="speech-2.8-turbo"
                )

        assert captured_args["voice_id"] == "Wise_Woman"
        assert captured_args["model"] == "speech-2.8-turbo"


@pytest.mark.unit
class TestMiniMaxTTSAdapterProcessText:
    """Tests for process_text method."""

    def test_process_text_returns_tensor(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "k"})

        fake_chunk = torch.randn(16000)

        with patch.object(adapter, "generate_single",
                          return_value=(fake_chunk, 32000)):
            result = adapter.process_text("Hello world", enable_chunking=False)

        assert isinstance(result, torch.Tensor)
        assert result.numel() > 0

    def test_process_text_with_return_info(self):
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        adapter = MiniMaxTTSAdapter({"api_key": "k"})

        with patch.object(adapter, "generate_single",
                          return_value=(torch.randn(16000), 32000)):
            result, info = adapter.process_text(
                "Hello", enable_chunking=False, return_info=True
            )

        assert isinstance(result, torch.Tensor)
        # info may be None for single-chunk text; verify the 2-tuple return shape
        assert isinstance(info, (dict, list, type(None)))


@pytest.mark.unit
class TestMiniMaxVoiceConstants:
    """Tests for voice and model constants."""

    def test_voice_ids_list(self):
        from engines.adapters.minimax_tts_adapter import MINIMAX_VOICE_IDS
        assert len(MINIMAX_VOICE_IDS) == 12
        assert "English_Graceful_Lady" in MINIMAX_VOICE_IDS
        assert "Deep_Voice_Man" in MINIMAX_VOICE_IDS
        assert "cute_boy" in MINIMAX_VOICE_IDS

    def test_models_list(self):
        from engines.adapters.minimax_tts_adapter import MINIMAX_MODELS
        assert "speech-2.8-hd" in MINIMAX_MODELS
        assert "speech-2.8-turbo" in MINIMAX_MODELS

    def test_voice_descriptions(self):
        from engines.adapters.minimax_tts_adapter import MINIMAX_VOICES
        assert MINIMAX_VOICES["English_Graceful_Lady"] == "Graceful Lady (English)"
        assert MINIMAX_VOICES["Deep_Voice_Man"] == "Deep Voice Man"


@pytest.mark.unit
class TestMiniMaxDecodeMp3:
    """Tests for MP3 decoding static method."""

    def test_decode_normalizes_multichannel(self):
        """Multi-channel audio should be averaged to mono."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        stereo = torch.randn(2, 16000)

        with patch("torchaudio.load", return_value=(stereo, 32000)):
            audio, sr = MiniMaxTTSAdapter._decode_mp3_to_tensor(b"\x00")

        assert audio.dim() == 1
        assert sr == 32000

    def test_decode_normalizes_amplitude(self):
        """Audio with values > 1.0 should be normalized to [-1, 1]."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        loud = torch.tensor([[5.0, -3.0, 2.0]])

        with patch("torchaudio.load", return_value=(loud, 32000)):
            audio, sr = MiniMaxTTSAdapter._decode_mp3_to_tensor(b"\x00")

        assert audio.abs().max() <= 1.0

    def test_decode_keeps_quiet_audio(self):
        """Audio already in [-1, 1] range should not be modified."""
        from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter

        quiet = torch.tensor([[0.5, -0.3, 0.1]])

        with patch("torchaudio.load", return_value=(quiet, 32000)):
            audio, sr = MiniMaxTTSAdapter._decode_mp3_to_tensor(b"\x00")

        assert torch.allclose(audio, quiet.squeeze(0))
