from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.voice.character_logging import (
    format_character_override_warning,
    format_resolved_character_block,
    format_resolved_character_text,
    resolved_character_label,
    resolved_voice_name,
)


@pytest.mark.unit
def test_resolved_voice_name_prefers_connected_character_name():
    voice_ref = {
        "character_name": "Tony",
        "audio_path": r"J:\voices\other-name.mp3",
    }

    assert resolved_voice_name(voice_ref) == "Tony"
    assert resolved_character_label("Bob", voice_ref) == "Tony"


@pytest.mark.unit
def test_resolved_voice_name_supports_audio_paths_and_legacy_tuples():
    audio_path = Path("voices") / "Ana_1.wav"

    assert resolved_voice_name({"audio_path": audio_path}) == "Ana_1"
    assert resolved_voice_name((str(audio_path), "reference text")) == "Ana_1"


@pytest.mark.unit
def test_resolved_voice_logging_formats_are_shared():
    voice_ref = {"character_name": "Tony"}

    assert format_resolved_character_text("Bob", "Hello!", voice_ref) == "[Tony] Hello!"
    assert format_resolved_character_block("Bob", "Hello!", voice_ref) == (
        "============================================================\n"
        "[Tony] Hello!\n"
        "============================================================"
    )
    assert format_character_override_warning("Fish", "Speaker 2", "Bob") == (
        "⚠️ Fish priority: Speaker 2 input overrides ['Bob'] alias"
    )
