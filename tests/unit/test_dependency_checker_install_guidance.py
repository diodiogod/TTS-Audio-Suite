from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.system.dependency_checker import DependencyChecker


@pytest.mark.unit
def test_install_guidance_uses_suite_installer_and_active_python():
    command = DependencyChecker.get_install_command()

    assert sys.executable in command
    assert str(REPO_ROOT / "install.py") in command
    assert "requirements.txt" not in command


@pytest.mark.unit
def test_startup_warning_gives_layman_repair_steps(monkeypatch):
    monkeypatch.setattr(
        DependencyChecker,
        "check_core_dependencies",
        staticmethod(lambda: []),
    )
    monkeypatch.setattr(
        DependencyChecker,
        "check_engine_dependencies",
        staticmethod(
            lambda engine: [("cached_path", "cached-path")] if engine == "f5tts" else []
        ),
    )

    warning = "\n".join(DependencyChecker.get_startup_warnings())

    assert "Some TTS Audio Suite engines are not installed correctly" in warning
    assert "1. Close ComfyUI." in warning
    assert "2. Open a terminal." in warning
    assert "3. Run this command:" in warning
    assert DependencyChecker.get_install_command() in warning
    assert "4. Start ComfyUI again." in warning
    assert "import: cached_path" not in warning
