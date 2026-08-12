import io
from pathlib import Path
import sys
import urllib.error

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp.catalog import load_catalog
from utils.audio_cpp.downloader import AudioCppDownloadError, install_package
from utils.audio_cpp.discovery import package_install_path


class FakeResponse(io.BytesIO):
    def __init__(self, payload, content_length=None):
        super().__init__(payload)
        self.status = 200
        self.headers = {
            "Content-Length": str(len(payload) if content_length is None else content_length)
        }


@pytest.mark.unit
def test_direct_hf_download_uses_auth_staging_and_nested_atomic_publish(tmp_path):
    package = load_catalog().package("pocket_tts_english_q8_0")
    payload = b"complete-gguf"
    requests = []

    def opener(request, timeout):
        requests.append((request, timeout))
        return FakeResponse(payload)

    result = install_package(package, tmp_path, token="secret-token", opener=opener)
    target = package_install_path(package, tmp_path)

    assert result.path == target
    assert (target / package.local_files[0]).read_bytes() == payload
    assert requests[0][0].get_header("Authorization") == "Bearer secret-token"
    assert "huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/" in requests[0][0].full_url
    assert not list(target.parent.glob("*.staging"))


@pytest.mark.unit
def test_incomplete_http_response_never_publishes_package(tmp_path):
    package = load_catalog().package("chatterbox_q8_0")

    def opener(request, timeout):
        return FakeResponse(b"short", content_length=100)

    with pytest.raises(AudioCppDownloadError, match="Incomplete download"):
        install_package(package, tmp_path, opener=opener)

    assert not package_install_path(package, tmp_path).exists()


@pytest.mark.unit
def test_install_preserves_sibling_precision_in_shared_target(tmp_path):
    package = load_catalog().package("chatterbox_q8_0")
    sibling_package = load_catalog().package("chatterbox_f16")
    target = package_install_path(package, tmp_path)
    sibling = package_install_path(sibling_package, tmp_path)
    target.mkdir(parents=True)
    sibling_file = sibling / sibling_package.local_files[0]
    sibling_file.write_bytes(b"keep")

    result = install_package(
        package,
        tmp_path,
        opener=lambda request, timeout: FakeResponse(b"new"),
    )

    assert (result.path / package.local_files[0]).read_bytes() == b"new"
    assert sibling_file.read_bytes() == b"keep"
    assert not list(target.parent.glob("*.backup"))


@pytest.mark.unit
def test_hf_auth_failure_is_actionable_and_leaves_no_target(tmp_path):
    package = load_catalog().package("chatterbox_q8_0")

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    with pytest.raises(AudioCppDownloadError, match="HF_TOKEN"):
        install_package(package, tmp_path, opener=opener)
    assert not package_install_path(package, tmp_path).exists()
