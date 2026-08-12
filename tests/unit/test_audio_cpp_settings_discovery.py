import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp.catalog import load_catalog
from utils.audio_cpp.discovery import (
    find_installed_package,
    package_install_path,
    resolve_model,
    resolve_model_roots,
)
from utils.audio_cpp.settings import AudioCppSettings, get_settings_path, load_settings, save_settings


class FakeFolderPaths:
    def __init__(self, user_root, models_root, registry):
        self.user_root = Path(user_root)
        self.models_dir = str(models_root)
        self.folder_names_and_paths = {
            key: ([str(path) for path in paths], set()) for key, paths in registry.items()
        }

    def get_system_user_directory(self, name):
        return str(self.user_root / name)

    def get_folder_paths(self, name):
        return list(self.folder_names_and_paths[name][0])


@pytest.mark.unit
def test_settings_live_under_comfyui_system_user_directory_and_round_trip(tmp_path):
    fake = FakeFolderPaths(tmp_path / "user", tmp_path / "models", {})
    expected = tmp_path / "user" / "tts_audio_suite" / "audio_cpp" / "settings.json"
    settings = AudioCppSettings(
        connection_mode="external",
        external_server_url="http://127.0.0.1:19090",
        model_roots=(str(tmp_path / "shared"),),
        runtime_backend="cuda",
        extras={"future_key": {"kept": True}},
    )

    assert get_settings_path(fake) == expected
    assert save_settings(settings, folder_paths_module=fake) == expected
    loaded = load_settings(folder_paths_module=fake, strict=True)
    assert loaded == settings
    assert json.loads(expected.read_text(encoding="utf-8"))["future_key"] == {"kept": True}
    assert not list(expected.parent.glob("*.tmp"))


@pytest.mark.unit
def test_broken_settings_fail_safe_unless_strict(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")

    assert load_settings(path) == AudioCppSettings()
    with pytest.raises(json.JSONDecodeError):
        load_settings(path, strict=True)


@pytest.mark.unit
def test_model_root_precedence_deduplicates_and_keeps_managed_last(tmp_path):
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    dedicated = tmp_path / "dedicated"
    tts_primary = tmp_path / "tts-primary"
    tts_secondary = tmp_path / "tts-secondary"
    managed = tts_primary / "audio.cpp" / "models"
    fake = FakeFolderPaths(
        tmp_path / "user",
        tmp_path / "models",
        {"audio_cpp": [dedicated], "TTS": [tts_primary, tts_secondary]},
    )
    settings = AudioCppSettings(
        model_roots=(str(configured), str(dedicated)),
        managed_model_root=str(managed),
    )

    assert resolve_model_roots(
        [explicit, managed], settings=settings, folder_paths_module=fake
    ) == [
        explicit,
        configured,
        dedicated,
        tts_secondary / "audio.cpp" / "models",
        managed,
    ]


@pytest.mark.unit
def test_discovery_prefers_existing_external_model_without_copying(tmp_path):
    package = load_catalog().package("chatterbox_q8_0")
    external = tmp_path / "existing-audio-cpp-models"
    managed = tmp_path / "managed"
    installed = package_install_path(package, external)
    installed.mkdir(parents=True)
    (installed / package.local_files[0]).write_bytes(b"gguf")

    assert find_installed_package(package, [external, managed]) == installed
    resolved = resolve_model(package.id, [external, managed])
    assert resolved is not None
    assert resolved.root == external
    assert resolved.path == installed
    assert not managed.exists()
