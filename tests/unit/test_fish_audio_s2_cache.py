from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio.cache import FishAudioS2CacheKeyGenerator
from engines.adapters.fish_audio_s2_adapter import FishAudioS2Adapter


@pytest.mark.unit
def test_quantization_variant_invalidates_audio_cache():
    generator = FishAudioS2CacheKeyGenerator()
    common = {
        "text": "Test",
        "audio_component": "voice-hash",
        "reference_text": "Reference",
        "seed": 1,
    }

    keys = {
        generator.generate_cache_key(**common, model_variant=variant)
        for variant in ("s2-pro", "s2-pro-fp8")
    }

    assert len(keys) == 2


@pytest.mark.unit
def test_load_quantization_invalidates_audio_cache():
    generator = FishAudioS2CacheKeyGenerator()
    common = {
        "text": "Test",
        "audio_component": "voice-hash",
        "reference_text": "Reference",
        "seed": 1,
        "model_variant": "s2-pro",
    }

    keys = {
        generator.generate_cache_key(**common, quantization=quantization)
        for quantization in ("none", "bnb_int8", "bnb_nf4")
    }

    assert len(keys) == 3


@pytest.mark.unit
def test_speaker_mode_invalidates_audio_cache():
    generator = FishAudioS2CacheKeyGenerator()
    common = {
        "text": "Test",
        "audio_component": "voice-hash",
        "reference_text": "Reference",
        "seed": 1,
    }

    native = generator.generate_cache_key(
        **common, multi_speaker_mode="Native Multi-Speaker"
    )
    custom = generator.generate_cache_key(
        **common, multi_speaker_mode="Custom Character Switching"
    )

    assert native != custom


@pytest.mark.unit
def test_adapter_reuses_unchanged_custom_segment_and_invalidates_changed_text(monkeypatch):
    adapter = FishAudioS2Adapter({"multi_speaker_mode": "Custom Character Switching"})
    adapter.audio_cache.clear_cache()
    calls = []

    class FakeEngine:
        def generate(self, **params):
            calls.append(params["text"])
            return torch.zeros(1, 32), adapter.SAMPLE_RATE

    monkeypatch.setattr(adapter, "_engine", lambda: FakeEngine())
    monkeypatch.setattr(adapter, "_reference", lambda _voice: ("voice.wav", "Reference", "voice-hash"))
    voice = {"audio_path": "voice.wav", "reference_text": "Reference"}

    adapter.generate_dialogue([(0, "unchanged")], [voice], seed=1, cache_character="speaker")
    adapter.generate_dialogue([(0, "unchanged")], [voice], seed=1, cache_character="speaker")
    adapter.generate_dialogue([(0, "changed")], [voice], seed=1, cache_character="speaker")

    assert len(calls) == 2
    adapter.audio_cache.clear_cache()
