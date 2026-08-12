import base64
import io
import json
import os
import struct
import sys
import threading
import types
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.audio_cpp.client import (
    AudioCppClient,
    AudioCppHTTPError,
    AudioCppProtocolError,
)
from utils.audio_cpp.catalog import load_catalog
from utils.audio_cpp.discovery import package_install_path
from utils.audio_cpp.process import AudioCppServerProcess, normalize_audio_cpp_task
from utils.audio_cpp.settings import AudioCppSettings
from utils.audio_cpp import resolver as audio_cpp_resolver
from utils.audio_cpp.session import (
    close_all_audio_cpp_sessions,
    get_audio_cpp_session,
)


def _wav_bytes(sample_rate=16000, channels=1, frames=32):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        samples = []
        for index in range(frames):
            value = int(16000 * ((index % 4) - 1.5) / 1.5)
            samples.extend([value] * channels)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


class _FakeAudioCppHandler(BaseHTTPRequestHandler):
    server_version = "FakeAudioCpp/0.5.1"

    def log_message(self, format, *args):
        return None

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._json({"status": "ok", "models": 1, "features": ["unload_models"]})
        elif parsed.path == "/v1/models":
            self.server.model_queries += 1
            self._json({
                "object": "list",
                "data": [{
                    "id": self.server.model_id,
                    "object": "model",
                    "family": "pocket_tts",
                    "task": "tts",
                    "mode": "offline",
                }],
            })
        elif parsed.path == "/v1/audio/voices":
            self.server.last_voice_query = parse_qs(parsed.query)
            self._json({"voices": ["alba", "cosette"]})
        else:
            self._json({"error": {"message": "not found", "type": "not_found"}}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(payload)
        request = payload.get("request", {})
        text = request.get("text")
        if text == "http-error":
            self._json(
                {"error": {"message": "model is busy", "type": "server_busy"}},
                503,
            )
            return

        encoded = base64.b64encode(self.server.wav_bytes).decode("ascii")
        if text == "named-only":
            self._json({
                "named_audio_outputs": [{
                    "id": "speech",
                    "audio": encoded,
                    "sample_rate": 16000,
                    "channels": 1,
                }],
                "timing": {},
            })
        elif text == "ambiguous":
            self._json({
                "named_audio_outputs": [
                    {"id": "left", "audio": encoded},
                    {"id": "right", "audio": encoded},
                ]
            })
        else:
            self._json({
                "audio": encoded,
                "sample_rate": 16000,
                "channels": 1,
                "timing": {"wall_ms": 1.0},
            })


@pytest.fixture
def fake_audio_cpp_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAudioCppHandler)
    server.model_id = "pocket"
    server.wav_bytes = _wav_bytes()
    server.requests = []
    server.last_voice_query = None
    server.model_queries = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture(autouse=True)
def clean_audio_cpp_sessions():
    close_all_audio_cpp_sessions()
    yield
    close_all_audio_cpp_sessions()


@pytest.mark.unit
def test_client_health_models_voices_and_primary_audio(fake_audio_cpp_server):
    server, url = fake_audio_cpp_server
    client = AudioCppClient(url)

    assert client.health()["status"] == "ok"
    assert client.models()[0]["id"] == "pocket"
    assert client.voices("pocket") == ["alba", "cosette"]
    assert server.last_voice_query == {"model": ["pocket"]}
    assert client.supports_feature("unload_models") is True

    result = client.run_task("pocket", {"text": "hello", "seed": "42"})
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.waveform.shape == (1, 32)
    assert result.waveform.dtype == torch.float32
    assert result.waveform.device.type == "cpu"
    assert server.requests[-1] == {
        "model": "pocket",
        "request": {"text": "hello", "seed": "42"},
    }


@pytest.mark.unit
def test_client_selects_sole_named_audio_and_rejects_ambiguous_output(fake_audio_cpp_server):
    _, url = fake_audio_cpp_server
    client = AudioCppClient(url)

    result = client.run_task("pocket", {"text": "named-only"})
    assert result.sample_rate == 16000
    assert list(result.named_audio) == ["speech"]
    assert result.waveform.data_ptr() == result.named_audio["speech"].waveform.data_ptr()

    with pytest.raises(AudioCppProtocolError, match="multiple named audio"):
        client.run_task("pocket", {"text": "ambiguous"})


@pytest.mark.unit
def test_client_surfaces_structured_http_error(fake_audio_cpp_server):
    _, url = fake_audio_cpp_server
    client = AudioCppClient(url)

    with pytest.raises(AudioCppHTTPError) as captured:
        client.run_task("pocket", {"text": "http-error"})

    assert captured.value.status == 503
    assert captured.value.error_type == "server_busy"
    assert "model is busy" in str(captured.value)


def _write_fake_server_script(path: Path) -> Path:
    script = r'''
import argparse
import base64
import io
import json
import struct
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
with open(args.config, "r", encoding="utf-8") as handle:
    config = json.load(handle)
model_id = config["models"][0]["id"]

buffer = io.BytesIO()
with wave.open(buffer, "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)
    wav_file.writeframes(struct.pack("<16h", *range(16)))
encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return None
    def send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self.send_json({"status": "ok", "models": 1})
        elif path == "/v1/models":
            self.send_json({"object": "list", "data": [{"id": model_id}]})
        elif path == "/v1/audio/voices":
            self.send_json({"voices": ["managed"]})
        else:
            self.send_json({})
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length).decode("utf-8"))
        self.send_json({"audio": encoded, "sample_rate": 22050, "channels": 1})

server = ThreadingHTTPServer((config["host"], config["port"]), Handler)
print("fake audio.cpp ready", flush=True)
server.serve_forever()
'''
    path.write_text(script, encoding="utf-8")
    return path


def _owned_config(tmp_path: Path, script: Path):
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    (model_path / "weights.gguf").write_bytes(b"audio-cpp-test-model")
    return {
        "connection_mode": "existing_binary",
        "binary_path": sys.executable,
        "binary_args": [str(script)],
        "model_path": str(model_path),
        "model_id": "managed-pocket",
        "family": "pocket_tts",
        "package_id": "pocket_tts_english_q8_0",
        "task": "tts",
        "backend": "cpu",
        "startup_timeout": 5.0,
        "connect_timeout": 1.0,
        "request_timeout": 5.0,
        "stop_timeout": 2.0,
    }


@pytest.mark.unit
def test_owned_process_writes_secure_config_and_stops_exact_child(tmp_path):
    assert normalize_audio_cpp_task("clone") == "clon"
    assert normalize_audio_cpp_task("voice design") == "vdes"
    script = _write_fake_server_script(tmp_path / "fake_audio_cpp_server.py")
    runtime = AudioCppServerProcess(_owned_config(tmp_path, script)).start()
    process = runtime.process
    config_path = runtime.config_path
    assert process is not None and process.poll() is None
    assert config_path is not None and config_path.exists()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["host"] == "127.0.0.1"
    assert payload["cors_origins"] == ""
    assert payload["log_request_body"] is False
    assert payload["model_spec_override"].endswith("utils\\audio_cpp\\model_specs") or payload[
        "model_spec_override"
    ].endswith("utils/audio_cpp/model_specs")
    assert payload["models"][0]["task"] == "tts"
    assert Path(payload["models"][0]["path"]).is_absolute()
    assert runtime.client.voices("managed-pocket") == ["managed"]

    runtime.close()
    assert process.poll() is not None
    assert not config_path.exists()


@pytest.mark.unit
def test_external_session_is_keyed_and_never_terminates_server(fake_audio_cpp_server):
    server, url = fake_audio_cpp_server
    config = {
        "connection_mode": "existing_server",
        "server_url": url,
        "model_id": "pocket",
    }
    first = get_audio_cpp_session(config)
    second = get_audio_cpp_session(dict(config))
    assert first is second
    assert first.owned is False
    assert first.model_id == "pocket"
    assert first.model_metadata["family"] == "pocket_tts"
    assert first.task == "tts"
    assert first.voices() == ["alba", "cosette"]
    assert server.model_queries == 1

    first.close()
    with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
        assert json.load(response)["status"] == "ok"


@pytest.mark.unit
def test_owned_session_restarts_and_reregisters_after_exact_child_exit(tmp_path, monkeypatch):
    script = _write_fake_server_script(tmp_path / "fake_audio_cpp_server.py")

    fake_management = types.ModuleType("comfy.model_management")
    fake_management.current_loaded_models = []
    fake_management.cleanup_models = lambda: None

    class LoadedModel:
        def __init__(self, model):
            self.model = model

    fake_management.LoadedModel = LoadedModel
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_management)
    comfy_module = sys.modules.get("comfy")
    if comfy_module is not None:
        monkeypatch.setattr(comfy_module, "model_management", fake_management, raising=False)

    session = get_audio_cpp_session(_owned_config(tmp_path, script))
    assert session.proxy.model_size() >= len(b"audio-cpp-test-model")
    first = session.run({"text": "first"})
    first_runtime = session.process
    first_process = first_runtime.process
    assert first.sample_rate == 22050
    assert len(fake_management.current_loaded_models) == 1

    first_process.kill()
    first_process.wait(timeout=2)
    second = session.run({"text": "second"})
    second_runtime = session.process
    second_process = second_runtime.process
    assert second.sample_rate == 22050
    assert second_runtime is not first_runtime
    assert second_process is not first_process
    assert len(fake_management.current_loaded_models) == 1

    assert session.proxy.partially_unload("cpu", 1) == 0
    tracked_model = fake_management.current_loaded_models[0]
    session.proxy.unpatch_model("cpu")
    assert second_process.poll() is not None
    assert fake_management.current_loaded_models == [tracked_model]
    fake_management.current_loaded_models.pop(0)

    session.run({"text": "third"})
    third_process = session.process.process
    assert third_process.poll() is None
    assert len(fake_management.current_loaded_models) == 1
    session.close()
    assert third_process.poll() is not None
    assert len(fake_management.current_loaded_models) == 0


@pytest.mark.unit
def test_session_makes_voice_reference_path_absolute(fake_audio_cpp_server, tmp_path, monkeypatch):
    server, url = fake_audio_cpp_server
    monkeypatch.chdir(tmp_path)
    reference = tmp_path / "voice.wav"
    reference.write_bytes(_wav_bytes())
    session = get_audio_cpp_session({
        "connection_mode": "existing_server",
        "server_url": url,
        "model_id": "pocket",
    })

    session.run({"text": "absolute path", "voice_ref": "voice.wav"})

    sent_path = server.requests[-1]["request"]["voice_ref"]
    assert Path(sent_path).is_absolute()
    assert Path(sent_path) == reference


@pytest.mark.unit
def test_session_resolver_discovers_existing_model_root(tmp_path, monkeypatch):
    script = _write_fake_server_script(tmp_path / "fake_audio_cpp_server.py")
    external_root = tmp_path / "existing-models"
    managed_root = tmp_path / "suite-managed-models"
    package = load_catalog().package("pocket_tts_english_q8_0")
    installed = package_install_path(package, external_root)
    for relative_path in package.local_files:
        target = installed / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"installed")

    monkeypatch.setattr(
        audio_cpp_resolver,
        "load_settings",
        lambda: AudioCppSettings(
            connection_mode="managed",
            model_roots=(str(external_root),),
            managed_model_root=str(managed_root),
            runtime_backend="cpu",
        ),
    )
    session = get_audio_cpp_session({
        "connection_mode": "managed",
        "family": "pocket_tts",
        "package_id": package.id,
        "task": "tts",
        "backend": "cpu",
        "binary_path": sys.executable,
        "binary_args": [str(script)],
        "startup_timeout": 5.0,
        "connect_timeout": 1.0,
        "request_timeout": 5.0,
    })

    result = session.run({"text": "resolved"})

    assert result.sample_rate == 22050
    assert session.config["model_path"] == str(installed.resolve())
    assert not managed_root.exists()


@pytest.mark.unit
def test_external_session_selects_the_servers_sole_model(fake_audio_cpp_server):
    server, url = fake_audio_cpp_server

    session = get_audio_cpp_session({
        "connection_mode": "external_server",
        "server_url": url,
        "model_id": "",
    })
    reused = get_audio_cpp_session({
        "connection_mode": "external_server",
        "server_url": url,
        "model_id": "",
    })

    assert reused is session
    assert session.model_id == "pocket"
    assert session.model_metadata["family"] == "pocket_tts"
    assert session.task == "tts"
    assert server.model_queries == 1
