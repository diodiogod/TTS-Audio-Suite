import pytest

from utils.audio.cache import FishAudioS2CacheKeyGenerator


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
