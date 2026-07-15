import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INSTALL_SPEC = importlib.util.spec_from_file_location("tts_suite_install_test_module", REPO_ROOT / "install.py")
INSTALL_MODULE = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(INSTALL_MODULE)

from engines.dots_tts.dots_tts_engine import DotsTTSEngine
from utils.runtimes.profiles import get_runtime_profile


@pytest.mark.unit
def test_fish_source_restore_accepts_namespace_package(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_package = source_root / "fish_speech"
    (source_package / "inference_engine").mkdir(parents=True)
    (source_package / "inference_engine" / "__init__.py").write_text("", encoding="utf-8")

    site_root = tmp_path / "site-packages"
    target_package = site_root / "fish_speech"
    target_package.mkdir(parents=True)
    (target_package / "content_sequence.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(site_root)])

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer._restore_fish_source_package(source_root) is True
    assert (target_package / "inference_engine" / "__init__.py").is_file()


@pytest.mark.unit
def test_dots_fallback_overrides_incomplete_tn_module_temporarily(monkeypatch):
    unrelated_tn = types.ModuleType("tn")
    monkeypatch.setitem(sys.modules, "tn", unrelated_tn)
    for name in (
        "tn.chinese",
        "tn.chinese.normalizer",
        "tn.english",
        "tn.english.normalizer",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with DotsTTSEngine._text_normalizer_compat():
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer

        assert ZhNormalizer().normalize("123") == "123"
        assert EnNormalizer().normalize("test") == "test"

    assert sys.modules["tn"] is unrelated_tn
    assert "tn.chinese" not in sys.modules
    assert "tn.english" not in sys.modules


@pytest.mark.unit
def test_vibevoice_runtime_avoids_webrtc_dependency_conflict():
    profile = get_runtime_profile("vibevoice_transformers4_shared")

    assert profile is not None
    assert "av" in profile.pip_packages
    assert not {
        "aiortc",
        "pyee",
        "dnspython",
        "ifaddr",
        "pylibsrtp",
        "pyopenssl",
    }.intersection(profile.pip_packages)
