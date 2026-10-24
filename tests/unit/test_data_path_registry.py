"""Unrelated custom-node folder metadata must not disable data path checks."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest


SPEC = importlib.util.spec_from_file_location(
    "data_path_registry_test_module",
    Path(__file__).resolve().parents[2] / "utils" / "security" / "path_access.py",
)
path_access = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(path_access)

pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch, tmp_path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(tmp_path / "models")
    folder_paths.get_input_directory = lambda: str(tmp_path / "input")
    folder_paths.get_output_directory = lambda: str(tmp_path / "output")
    folder_paths.get_temp_directory = lambda: str(tmp_path / "temp")
    folder_paths.folder_names_and_paths = {
        "voices": ([str(tmp_path / "shared_voices")], {".wav"}),
        "custom_nodes": ([str(tmp_path / "custom_nodes")], set()),
    }
    # Match ComfyUI's public accessor, which reads paths without unpacking metadata.
    folder_paths.get_folder_paths = lambda name: folder_paths.folder_names_and_paths[name][0][:]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    return folder_paths


@pytest.mark.parametrize(
    "entry",
    [
        ([], set(), "extra metadata"),
        (),
        None,
        "not a registry entry",
        {"paths": []},
        (["valid-looking"],),
        ("a single path string", set()),
        ([None, 123], set()),
    ],
)
def test_unrelated_entries_preserve_voice_model_and_security_checks(registry, tmp_path, entry):
    registry.folder_names_and_paths["unrelated_extension"] = entry
    for relative in ("models/OmniVoice/model.pt", "shared_voices/narrator.wav", "input/audio.wav"):
        file_path = tmp_path / relative
        assert path_access.allowed_path(file_path) == str(file_path.resolve())

    bundled_voice = Path(path_access.__file__).resolve().parents[2] / "voices_examples" / "female" / "voice.wav"
    assert path_access.allowed_path(bundled_voice) == str(bundled_voice)

    for relative in ("outside/secret.txt", "custom_nodes/code.py", "shared_voices/../outside/secret.txt"):
        with pytest.raises(ValueError, match="outside the permitted"):
            path_access.allowed_path(tmp_path / relative)


def test_extra_metadata_retains_registered_external_paths(registry, tmp_path):
    external = tmp_path / "external_models"
    registry.folder_names_and_paths["extended_model_type"] = ([str(external)], set(), "extra metadata")
    assert path_access.allowed_path(external / "model.pt") == str(external / "model.pt")


def test_excluded_custom_nodes_entry_is_not_parsed(registry, tmp_path):
    registry.folder_names_and_paths["custom_nodes"] = ([str(tmp_path / "custom_nodes")], set(), "extra metadata")
    assert path_access.allowed_path(tmp_path / "models" / "voice.wav")
    with pytest.raises(ValueError, match="outside the permitted"):
        path_access.allowed_path(tmp_path / "custom_nodes" / "code.py")


def test_models_false_does_not_allow_model_or_voice_roots(registry, tmp_path):
    registry.folder_names_and_paths["unrelated_extension"] = ([], set(), "extra metadata")
    assert path_access.allowed_path(tmp_path / "input" / "audio.wav", models=False)
    for relative in ("models/model.pt", "shared_voices/voice.wav"):
        with pytest.raises(ValueError, match="outside the permitted"):
            path_access.allowed_path(tmp_path / relative, models=False)
