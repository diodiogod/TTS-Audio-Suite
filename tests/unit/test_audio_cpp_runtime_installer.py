import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp.runtime_installer import (
    RuntimeAsset,
    RuntimeInstallError,
    RuntimeManifest,
    get_runtime_manifest,
    install_windows_runtime,
    runtime_install_path,
)


class FakeResponse(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.status = 200
        self.headers = {"Content-Length": str(len(payload))}


def make_zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for name, payload in files.items():
            bundle.writestr(name, payload)
    return stream.getvalue()


def fake_manifest(backend, payloads, required):
    assets = tuple(
        RuntimeAsset(
            filename=name,
            url=f"https://example.test/{name}",
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for name, payload in payloads.items()
    )
    return RuntimeManifest(backend=backend, assets=assets, required_files=tuple(required))


@pytest.mark.unit
def test_pinned_windows_manifests_include_verified_cuda_runtime_asset():
    cpu = get_runtime_manifest("cpu")
    cuda = get_runtime_manifest("cuda")

    assert len(cpu.assets) == 1
    assert len(cuda.assets) == 2
    assert cuda.assets[1].filename == "audiocpp-windows-cuda-runtime.zip"
    assert cuda.assets[1].sha256 == (
        "46016655aff8f050806d81efd0fe256c15b86527935bfb3896208d4cac6b5ff8"
    )
    assert "cublas64_13.dll" in cuda.required_files


@pytest.mark.unit
def test_cuda_runtime_installs_two_verified_archives_atomically(tmp_path):
    executable_zip = make_zip(
        {"audiocpp_server.exe": b"server", "audiocpp_cli.exe": b"cli"}
    )
    cuda_zip = make_zip({"cublas64_13.dll": b"cublas", "cufft64_12.dll": b"cufft"})
    payloads = {"runtime.zip": executable_zip, "cuda.zip": cuda_zip}
    manifest = fake_manifest(
        "cuda",
        payloads,
        ("audiocpp_server.exe", "audiocpp_cli.exe", "cublas64_13.dll", "cufft64_12.dll"),
    )

    def opener(request, timeout):
        return FakeResponse(payloads[Path(request.full_url).name])

    result = install_windows_runtime(
        tmp_path,
        "cuda",
        manifest=manifest,
        platform_name="win32",
        opener=opener,
    )

    assert result.path == runtime_install_path(tmp_path, "cuda")
    assert result.executable.read_bytes() == b"server"
    assert (result.path / "cublas64_13.dll").read_bytes() == b"cublas"
    assert not list(result.path.parent.glob("*.staging"))


@pytest.mark.unit
def test_runtime_hash_failure_does_not_publish_or_destroy_existing_target(tmp_path):
    archive = make_zip({"audiocpp_server.exe": b"new", "audiocpp_cli.exe": b"cli"})
    asset = RuntimeAsset(
        filename="runtime.zip",
        url="https://example.test/runtime.zip",
        size=len(archive),
        sha256="0" * 64,
    )
    manifest = RuntimeManifest(
        backend="cpu",
        assets=(asset,),
        required_files=("audiocpp_server.exe", "audiocpp_cli.exe"),
    )
    target = runtime_install_path(tmp_path, "cpu")
    target.mkdir(parents=True)
    marker = target / "old.txt"
    marker.write_bytes(b"old")

    with pytest.raises(RuntimeInstallError, match="SHA256 mismatch"):
        install_windows_runtime(
            tmp_path,
            "cpu",
            overwrite=True,
            manifest=manifest,
            platform_name="win32",
            opener=lambda request, timeout: FakeResponse(archive),
        )

    assert marker.read_bytes() == b"old"
    assert not (target / "audiocpp_server.exe").exists()


@pytest.mark.unit
def test_runtime_rejects_archive_path_traversal(tmp_path):
    archive = make_zip(
        {"../escape.dll": b"bad", "audiocpp_server.exe": b"server", "audiocpp_cli.exe": b"cli"}
    )
    manifest = fake_manifest(
        "cpu", {"runtime.zip": archive}, ("audiocpp_server.exe", "audiocpp_cli.exe")
    )

    with pytest.raises(RuntimeInstallError, match="Unsafe path"):
        install_windows_runtime(
            tmp_path,
            "cpu",
            manifest=manifest,
            platform_name="win32",
            opener=lambda request, timeout: FakeResponse(archive),
        )
    assert not (tmp_path / "escape.dll").exists()
