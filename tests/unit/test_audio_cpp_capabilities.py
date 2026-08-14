import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp.capabilities import (
    get_package_dependencies,
    get_capability,
    load_capabilities,
    public_capabilities,
    validate_voice_reference,
)
from utils.audio_cpp.catalog import load_catalog


@pytest.mark.unit
def test_capability_overlay_covers_the_pinned_catalog():
    capabilities = load_capabilities()
    assert set(capabilities) == set(load_catalog().families)
    assert capabilities["vibevoice"]["native_multi_speaker"] == {
        "supported": True,
        "max_speakers": 4,
        "suite_status": "partial",
    }
    assert capabilities["vibevoice_asr"]["asr_features"] == {
        "diarization": "native",
        "timing": "native_segment",
    }
    assert capabilities["nemotron_asr"]["asr_features"] == {
        "diarization": "none",
        "timing": "native_word",
    }
    assert capabilities["qwen3_asr"]["asr_features"]["timing"] == "optional_forced_aligner"
    assert capabilities["voxtral_realtime"]["asr_features"] == {
        "diarization": "none",
        "timing": "none",
    }
    public = public_capabilities()
    assert set(public["packages"]) == set(load_catalog().packages)
    assert public["packages"]["qwen3_tts_1_7b_base_q8_0"]["estimated_download_bytes"] == 2695175104
    mio = public["packages"]["miotts_1_7b_q8_0"]
    assert mio["dependencies"] == ["miocodec_q8_0"]
    assert mio["estimated_download_bytes"] == 2496393216
    assert get_package_dependencies("miotts_1_7b_q8_0")[0]["session_option"] == "miotts.codec_model_path"


@pytest.mark.unit
def test_glm_requires_audio_and_matching_transcript():
    with pytest.raises(ValueError, match="requires reference audio"):
        validate_voice_reference("glm_tts", {}, "Alice")
    with pytest.raises(ValueError, match="requires the transcript"):
        validate_voice_reference(
            "glm_tts",
            {"audio": {"waveform": object(), "sample_rate": 24000}},
            "Alice",
        )


@pytest.mark.unit
def test_optional_reference_family_accepts_default_voice():
    validate_voice_reference("pocket_tts", {}, "narrator")
    assert get_capability("supertonic")["built_in_voices"] is True
