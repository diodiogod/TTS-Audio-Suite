import importlib.util
import json
import shutil
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INSTALL_SPEC = importlib.util.spec_from_file_location("tts_suite_install_test_module", REPO_ROOT / "install.py")
INSTALL_MODULE = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(INSTALL_MODULE)

from engines.dots_tts.dots_tts_engine import DotsTTSEngine
from utils.runtimes.bootstrap import BOOTSTRAP_STRATEGY
from utils.runtimes.profiles import get_runtime_profile


@pytest.mark.unit
@pytest.mark.parametrize("shadow_utils", ["absent", "package", "cached_package", "module"])
@pytest.mark.parametrize("install_shared_runtime", [True, False])
def test_installer_runtime_helpers_work_with_isolated_import_paths(
    tmp_path, shadow_utils, install_shared_runtime
):
    suite = tmp_path / "ComfyUI" / "custom_nodes" / "tts_audio_suite"
    suite.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "install.py", suite / "install.py")
    shutil.copytree(
        REPO_ROOT / "utils" / "runtimes", suite / "utils" / "runtimes",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # A prepared runtime must be reusable without importing the engine stack
    # or touching pip. Missing preferences intentionally exercise default-on.
    runtime = suite / "runtimes" / "shared_legacy_t4"
    python_path = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(b"prepared runtime sentinel")
    (runtime / "runtime_metadata.json").write_text(json.dumps({
        "profile": asdict(get_runtime_profile("vibevoice_transformers4_shared")),
        "bootstrap_strategy": BOOTSTRAP_STRATEGY,
        "install_complete": True,
        "source_python": sys.executable,
    }), encoding="utf-8")
    if not install_shared_runtime:
        settings = tmp_path / "ComfyUI" / "user" / "__tts_audio_suite" / "runtime_settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text('{"install_shared_runtime": false}', encoding="utf-8")

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    if shadow_utils in ("package", "cached_package"):
        (foreign / "utils").mkdir()
        (foreign / "utils" / "__init__.py").write_text("", encoding="utf-8")
    elif shadow_utils == "module":
        (foreign / "utils.py").write_text("", encoding="utf-8")

    code = """
import os, runpy, subprocess, sys
from pathlib import Path
suite, foreign, shadow_utils = sys.argv[1:]
os.environ['COMFYUI_PATH'] = str(Path(suite).parents[1])
os.environ.pop('TTS_AUDIO_SUITE_USER_DIRECTORY', None)
sys.path.insert(0, foreign)
if shadow_utils in ('cached_package', 'module'):
    import utils
original_utils = sys.modules.get('utils')
original_path = sys.path[:]
assert suite not in sys.path
def unexpected_subprocess(*args, **kwargs):
    raise AssertionError('A prepared or opted-out runtime must not run installers')
subprocess.run = unexpected_subprocess
installer = runpy.run_path(str(Path(suite) / 'install.py'), run_name='isolated_install')['TTSAudioInstaller']()
installer.prepare_shared_runtime()
installer.prepare_shared_runtime()
assert sys.modules.get('utils') is original_utils
assert sys.path == original_path
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(suite), str(foreign), shadow_utils],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    expected = "Preparing Shared Runtime" if install_shared_runtime else "Shared Runtime installation skipped"
    assert expected in result.stdout
    assert python_path.read_bytes() == b"prepared runtime sentinel"


@pytest.mark.unit
def test_nested_module_check_does_not_import_parent(tmp_path, monkeypatch):
    package = tmp_path / "explosive_parent"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('parent package must not be imported')\n",
        encoding="utf-8",
    )
    (package / "runtime.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "explosive_parent", raising=False)

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer.module_available("explosive_parent.runtime") is True
    assert installer.module_available("explosive_parent.missing") is False
    assert "explosive_parent" not in sys.modules


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile_name",
    ("vibevoice_transformers4_shared", "qwen3_tts_transformers4_dedicated"),
)
def test_legacy_transformers_runtime_pins_compatible_kernels(profile_name):
    profile = get_runtime_profile(profile_name)

    assert profile is not None
    assert "kernels>=0.6.1,<=0.9" in profile.pip_packages
