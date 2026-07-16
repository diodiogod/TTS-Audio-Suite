import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.models.factory_config import ModelLoadConfig
from utils.runtimes import vibevoice_proxy


@pytest.mark.unit
def test_isolated_proxy_receives_parent_resolved_local_model_path(monkeypatch):
    expected_path = r"C:\ComfyUI\models\TTS\vibevoice\vibevoice-7B-Parker"

    class FakeDownloader:
        def get_model_path(self, model_name):
            assert model_name == "vibevoice-7B-Parker"
            return expected_path

    downloader_module = types.ModuleType(
        "engines.vibevoice_engine.vibevoice_downloader"
    )
    downloader_module.VibeVoiceDownloader = FakeDownloader
    monkeypatch.setitem(
        sys.modules,
        "engines.vibevoice_engine.vibevoice_downloader",
        downloader_module,
    )
    monkeypatch.setattr(
        vibevoice_proxy,
        "VibeVoiceIsolatedProxy",
        lambda config, profile: types.SimpleNamespace(config=config, profile=profile),
    )

    config = ModelLoadConfig(
        engine_name="vibevoice",
        model_type="tts",
        model_name="local:vibevoice-7B-Parker",
        device="cpu",
        runtime_mode="shared_runtime",
        runtime_profile="vibevoice_transformers4_shared",
    )

    proxy = vibevoice_proxy.build_vibevoice_isolated_proxy(config)

    assert config.model_path == expected_path
    assert proxy.config.model_path == expected_path
