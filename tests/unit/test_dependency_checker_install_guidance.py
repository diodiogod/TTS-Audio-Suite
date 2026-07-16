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
