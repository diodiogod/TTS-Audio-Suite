from pathlib import Path
import importlib.util
import sys
import types

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.audio.processing import AudioProcessingUtils


folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(REPO_ROOT / "models")
sys.modules.setdefault("folder_paths", folder_paths)

NODE_SPEC = importlib.util.spec_from_file_location(
    "character_voices_node_test_module",
    REPO_ROOT / "nodes" / "shared" / "character_voices_node.py",
)
character_voices_node = importlib.util.module_from_spec(NODE_SPEC)
NODE_SPEC.loader.exec_module(character_voices_node)


@pytest.mark.unit
def test_legacy_empty_transcription_restores_library_reference(monkeypatch, tmp_path):
    audio_path = tmp_path / "Legacy Voice.wav"
    audio_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        character_voices_node,
        "load_voice_reference",
        lambda _voice_name: (str(audio_path), "Canonical spoken transcription."),
    )
    monkeypatch.setattr(
        AudioProcessingUtils,
        "safe_load_audio",
        staticmethod(lambda _path: (torch.zeros(1, 2400), 24000)),
    )

    voice, character_name, _ = character_voices_node.CharacterVoicesNode().get_voice_reference(
        voice_name="Legacy Voice.wav",
        reference_text="",
        customized=True,
    )

    assert character_name == "Legacy Voice"
    assert voice["reference_text"] == "Canonical spoken transcription."
    assert voice["canonical_reference_text"] == "Canonical spoken transcription."
