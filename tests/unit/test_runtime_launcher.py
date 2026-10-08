from pathlib import Path
import os
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
@pytest.mark.parametrize("root_present", [False, True])
@pytest.mark.parametrize("profile_override", [False, True])
def test_build_env_prioritizes_suite_over_bundled_modules(monkeypatch, tmp_path, root_present, profile_override):
    bundled = str(REPO_ROOT / "engines" / "step_audio_editx" / "step_audio_editx_impl")
    comfy = str(tmp_path / "ComfyUI")
    site_packages = str(tmp_path / "site-packages")
    parent_paths = [bundled, comfy, site_packages]
    if root_present:
        parent_paths += [str(REPO_ROOT), str(REPO_ROOT / "utils" / "..")]
    monkeypatch.setattr(sys, "path", parent_paths.copy())
    inherited = os.pathsep.join([bundled, str(REPO_ROOT / "utils" / ".."), site_packages])
    monkeypatch.setenv("PYTHONPATH", inherited)
    monkeypatch.setattr(
        IsolatedRuntimeLauncher,
        "_augment_windows_toolchain_env",
        staticmethod(lambda env: env),
    )
    profile = RuntimeProfile(
        name="test", engine_names=[],
        env_vars={"PYTHONPATH": inherited} if profile_override else {},
    )

    # A separate runtime location must not replace the suite's source directory.
    env = IsolatedRuntimeLauncher(runtime_root=str(tmp_path)).build_env(profile)
    paths = env["PYTHONPATH"].split(os.pathsep)
    normalized = [os.path.normcase(os.path.realpath(path)) for path in paths]

    assert paths[0] == str(REPO_ROOT)
    assert len(normalized) == len(set(normalized))
    assert bundled in paths
    assert comfy in paths
    assert site_packages not in paths
    assert sys.path == parent_paths


@pytest.mark.unit
def test_build_env_preserves_relative_host_paths_when_worker_changes_cwd(monkeypatch, tmp_path):
    host = tmp_path / "host"
    dependency = host / "dependency"
    child = tmp_path / "child"
    dependency.mkdir(parents=True)
    child.mkdir()
    (host / "issue366_host_module.py").write_text("VALUE = 366\n")
    (dependency / "issue366_dependency.py").write_text("VALUE = 367\n")
    monkeypatch.chdir(host)
    monkeypatch.setattr(sys, "path", [".", str(host), "", str(REPO_ROOT)])
    monkeypatch.setenv("PYTHONPATH", "dependency")
    monkeypatch.setattr(IsolatedRuntimeLauncher, "_augment_windows_toolchain_env", staticmethod(lambda env: env))

    env = IsolatedRuntimeLauncher().build_env(RuntimeProfile(name="test", engine_names=[]))
    paths = env["PYTHONPATH"].split(os.pathsep)
    assert paths == [str(REPO_ROOT), str(host), str(dependency)]
    result = subprocess.run(
        [sys.executable, "-c", "import issue366_host_module as h, issue366_dependency as d; assert (h.VALUE, d.VALUE) == (366, 367)"],
        env=env, cwd=child, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("worker", sorted((REPO_ROOT / "utils/runtimes/workers").glob("*_worker.py")), ids=lambda path: path.stem)
def test_worker_bootstrap_ignores_shadow_utils(tmp_path, worker):
    (tmp_path / "utils.py").write_text('raise RuntimeError("Wrong utils imported")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(tmp_path),
        str(REPO_ROOT / "engines/step_audio_editx/step_audio_editx_impl"),
        str(REPO_ROOT),
    ])
    # Execute the real entrypoint's imports in a fresh interpreter without
    # entering its model loop. No model downloads or GPU allocation are needed.
    code = """
import runpy, sys
from pathlib import Path
worker, root = sys.argv[1:]
namespace = runpy.run_path(worker, run_name="worker_bootstrap_test")
import utils
assert Path(utils.__file__).resolve() == Path(root) / "utils/__init__.py"
assert Path(sys.path[0]).resolve() == Path(root)
assert namespace["RuntimeJobResponse"].__module__ == "utils.runtimes.protocol"
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(worker), str(REPO_ROOT)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_step_device_patch_uses_bundled_encoder_and_its_device(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    code = """
from utils.compatibility.step_audio_editx_device_patch import StepAudioEditXDevicePatches
StepAudioEditXDevicePatches.apply_all_patches(verbose=False)
from stepvocoder.cosyvoice2.transformer.upsample_encoder_v2 import UpsampleConformerEncoderV2
patched = UpsampleConformerEncoderV2._init_cuda_graph
assert getattr(patched, "_device_patched", False)
encoder = UpsampleConformerEncoderV2(input_size=8, output_size=8, num_blocks=1, num_up_blocks=1, attention_heads=2, linear_units=16)
encoder.enable_cuda_graph = True
encoder._init_cuda_graph()
assert encoder.enable_cuda_graph is False
StepAudioEditXDevicePatches.apply_all_patches(verbose=False)
assert UpsampleConformerEncoderV2._init_cuda_graph is patched
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
