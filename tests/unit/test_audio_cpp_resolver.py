from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp import resolver
from utils.audio_cpp.catalog import load_catalog
from utils.audio_cpp.discovery import package_install_path
from utils.audio_cpp.runtime_installer import runtime_install_path
from utils.audio_cpp.settings import AudioCppSettings


@pytest.mark.unit
def test_auto_reuses_machine_external_server_without_loading_owned_dependencies(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "load_settings",
        lambda: AudioCppSettings(
            connection_mode="external",
            external_server_url="HTTP://127.0.0.1:18080/",
            executable_path="C:/ignored/audiocpp_server.exe",
        ),
    )
    monkeypatch.setattr(
        resolver,
        "_catalog_module",
        lambda: (_ for _ in ()).throw(AssertionError("external mode loaded the catalog")),
    )
    monkeypatch.setattr(
        resolver,
        "_downloader_module",
        lambda: (_ for _ in ()).throw(AssertionError("external mode loaded a downloader")),
    )
    monkeypatch.setattr(
        resolver,
        "_runtime_installer_module",
        lambda: (_ for _ in ()).throw(AssertionError("external mode loaded an installer")),
    )

    result = resolver.resolve_audio_cpp_config(
        {
            "config": {
                "connection_mode": "auto",
                "family": "qwen3_tts",
                "package_id": "auto",
                "task": "auto",
            }
        }
    )

    assert result["connection_mode"] == "external_server"
    assert result["server_url"] == "http://127.0.0.1:18080"
    assert result["binary_path"] == ""
    assert result["model_path"] == ""
    assert "model_id" not in result
    assert result["task"] == "auto"


@pytest.mark.unit
def test_owned_explicit_paths_resolve_recommended_package_task_and_cuda(monkeypatch, tmp_path):
    binary = tmp_path / "audiocpp_server.exe"
    binary.write_bytes(b"exe")
    model = tmp_path / "Qwen-VoiceDesign"
    model.mkdir()
    monkeypatch.setattr(resolver, "load_settings", lambda: AudioCppSettings())
    monkeypatch.setattr(resolver, "_cuda_available", lambda: True)

    result = resolver.resolve_audio_cpp_config(
        {
            "connection_mode": "managed",
            "family": "qwen3_tts",
            "package_id": "qwen3_tts_1_7b_voicedesign_q8_0",
            "requested_task": "auto",
            "backend": "auto",
            "device": "cuda:2",
            "binary_path": str(binary),
            "model_path": str(model),
            "model_id": "My Qwen model",
        }
    )

    assert result["connection_mode"] == "owned_process"
    assert result["package_id"] == "qwen3_tts_1_7b_voicedesign_q8_0"
    assert result["task"] == "vdes"
    assert result["backend"] == "cuda"
    assert result["device_index"] == 2
    assert result["binary_path"] == str(binary.resolve())
    assert result["model_path"] == str(model.resolve())
    assert result["model_id"] == "My-Qwen-model"


@pytest.mark.unit
def test_owned_reuses_external_model_root_and_configured_runtime_root(monkeypatch, tmp_path):
    external_models = tmp_path / "existing-audio-cpp" / "models"
    managed_models = tmp_path / "suite" / "audio.cpp" / "models"
    configured_runtime = tmp_path / "existing-audio-cpp" / "runtime"
    settings = AudioCppSettings(
        connection_mode="managed",
        model_roots=(str(external_models),),
        managed_model_root=str(managed_models),
        runtime_root=str(configured_runtime),
        runtime_backend="cpu",
    )
    package = load_catalog().package("chatterbox_q8_0")
    installed_model = package_install_path(package, external_models)
    installed_model.mkdir(parents=True)
    (installed_model / package.local_files[0]).write_bytes(b"gguf")
    installed_binary = runtime_install_path(configured_runtime, "cpu") / "audiocpp_server.exe"
    installed_binary.parent.mkdir(parents=True)
    installed_binary.write_bytes(b"exe")
    monkeypatch.setattr(resolver, "load_settings", lambda: settings)

    result = resolver.resolve_audio_cpp_config(
        {
            "connection_mode": "auto",
            "family": "chatterbox",
            "package_id": "auto",
            "task": "auto",
            "backend": "auto",
            "auto_download_model": False,
            "auto_download_runtime": False,
        }
    )

    assert result["package_id"] == "chatterbox_q8_0"
    assert result["task"] == "clon"
    assert result["backend"] == "cpu"
    assert result["model_path"] == str(installed_model.resolve())
    assert result["binary_path"] == str(installed_binary.resolve())
    assert not managed_models.exists()


@pytest.mark.unit
def test_missing_assets_download_only_to_managed_roots(monkeypatch, tmp_path):
    external_models = tmp_path / "external" / "models"
    managed_models = tmp_path / "managed" / "audio.cpp" / "models"
    settings = AudioCppSettings(
        connection_mode="managed",
        model_roots=(str(external_models),),
        managed_model_root=str(managed_models),
        runtime_backend="cpu",
    )
    calls = {}
    real_downloader = resolver._downloader_module()
    real_runtime = resolver._runtime_installer_module()

    def install_package(package, root, catalog):
        calls["model_root"] = Path(root)
        target = package_install_path(package, root)
        target.mkdir(parents=True)
        (target / package.local_files[0]).write_bytes(b"gguf")
        return SimpleNamespace(path=target)

    def install_runtime(root, backend):
        calls["runtime_root"] = Path(root)
        calls["backend"] = backend
        executable = runtime_install_path(root, backend) / "audiocpp_server.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"exe")
        return SimpleNamespace(executable=executable)

    monkeypatch.setattr(resolver, "load_settings", lambda: settings)
    monkeypatch.setattr(
        resolver,
        "_downloader_module",
        lambda: SimpleNamespace(install_package=install_package),
    )
    monkeypatch.setattr(
        resolver,
        "_runtime_installer_module",
        lambda: SimpleNamespace(
            runtime_install_path=real_runtime.runtime_install_path,
            install_windows_runtime=install_runtime,
        ),
    )

    result = resolver.resolve_audio_cpp_config(
        {
            "connection_mode": "managed",
            "family": "chatterbox",
            "package_id": "chatterbox_q8_0",
            "backend": "cpu",
            "auto_download_model": True,
            "auto_download_runtime": True,
        }
    )

    assert calls["model_root"] == managed_models
    assert calls["runtime_root"] == managed_models.parent / "runtime"
    assert calls["backend"] == "cpu"
    assert not external_models.exists()
    assert result["model_path"].startswith(str(managed_models.resolve()))
    assert result["binary_path"].startswith(str((managed_models.parent / "runtime").resolve()))


@pytest.mark.unit
def test_missing_model_without_permission_has_actionable_error(monkeypatch, tmp_path):
    managed_models = tmp_path / "managed" / "models"
    binary = tmp_path / "audiocpp_server.exe"
    binary.write_bytes(b"exe")
    monkeypatch.setattr(
        resolver,
        "load_settings",
        lambda: AudioCppSettings(managed_model_root=str(managed_models)),
    )

    with pytest.raises(resolver.AudioCppResolutionError, match="enable auto_download_model"):
        resolver.resolve_audio_cpp_config(
            {
                "connection_mode": "managed",
                "family": "chatterbox",
                "package_id": "chatterbox_q8_0",
                "backend": "cpu",
                "binary_path": str(binary),
                "auto_download_model": False,
            }
        )


@pytest.mark.unit
def test_owned_existing_binary_allows_explicit_hip_backend(monkeypatch, tmp_path):
    binary = tmp_path / "audiocpp_server.exe"
    binary.write_bytes(b"exe")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(resolver, "load_settings", lambda: AudioCppSettings())

    result = resolver.resolve_audio_cpp_config(
        {
            "connection_mode": "existing_binary",
            "family": "chatterbox",
            "package_id": "chatterbox_q8_0",
            "backend": "hip",
            "binary_path": str(binary),
            "model_path": str(model),
        }
    )

    assert result["backend"] == "hip"
    assert result["binary_path"] == str(binary.resolve())


@pytest.mark.unit
def test_cpu_backend_reuses_installed_cuda_profile_before_downloading_duplicate(
    monkeypatch, tmp_path
):
    model = tmp_path / "model"
    model.mkdir()
    runtime_root = tmp_path / "runtime"
    cuda_binary = runtime_install_path(runtime_root, "cuda") / "audiocpp_server.exe"
    cuda_binary.parent.mkdir(parents=True)
    cuda_binary.write_bytes(b"cuda-exe")
    monkeypatch.setattr(
        resolver,
        "load_settings",
        lambda: AudioCppSettings(runtime_root=str(runtime_root), runtime_backend="cpu"),
    )

    result = resolver.resolve_audio_cpp_config(
        {
            "connection_mode": "managed",
            "family": "chatterbox",
            "package_id": "chatterbox_q8_0",
            "model_path": str(model),
            "backend": "auto",
            "auto_download_runtime": False,
        }
    )

    assert result["backend"] == "cpu"
    assert result["binary_path"] == str(cuda_binary.resolve())
