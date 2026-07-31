import json
import os
from pathlib import Path
import subprocess
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


@pytest.mark.unit
def test_audio8_worker_prioritizes_project_root_over_shadowing_utils(tmp_path):
    shadow_dir = tmp_path / "step_audio_editx_impl"
    shadow_dir.mkdir()
    (shadow_dir / "utils.py").write_text(
        'raise RuntimeError("shadow utils imported")\n',
        encoding="utf-8",
    )

    worker_path = REPO_ROOT / "utils" / "runtimes" / "workers" / "audio8_tts_worker.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(shadow_dir), str(REPO_ROOT)))
    completed = subprocess.run(
        [sys.executable, str(worker_path)],
        input=json.dumps(
            {
                "action": "ping",
                "request_id": "audio8-path-priority-test",
            }
        )
        + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout.strip())
    assert response["ok"] is True
    assert response["result"] == {"pong": True}
