from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.runtimes.launcher import IsolatedRuntimeLauncher
from utils.runtimes.profiles import RuntimeProfile


@pytest.mark.unit
@pytest.mark.parametrize("value", ["None", "-1", "4294967296", "not-a-seed"])
def test_build_env_removes_invalid_pythonhashseed(monkeypatch, value):
    monkeypatch.setenv("PYTHONHASHSEED", value)
    monkeypatch.setattr(
        IsolatedRuntimeLauncher,
        "_augment_windows_toolchain_env",
        staticmethod(lambda env: env),
    )

    profile = RuntimeProfile(name="test", engine_names=[])
    env = IsolatedRuntimeLauncher().build_env(profile)

    assert "PYTHONHASHSEED" not in env


@pytest.mark.unit
@pytest.mark.parametrize("value", ["random", "0", "123", "4294967295"])
def test_build_env_preserves_valid_pythonhashseed(monkeypatch, value):
    monkeypatch.setenv("PYTHONHASHSEED", value)
    monkeypatch.setattr(
        IsolatedRuntimeLauncher,
        "_augment_windows_toolchain_env",
        staticmethod(lambda env: env),
    )

    profile = RuntimeProfile(name="test", engine_names=[])
    env = IsolatedRuntimeLauncher().build_env(profile)

    assert env["PYTHONHASHSEED"] == value
