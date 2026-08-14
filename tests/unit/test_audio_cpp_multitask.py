"""No-model tests for audio.cpp ASR and unified voice-conversion contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.asr.types import ASRRequest
from utils.audio_cpp import session as session_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asr_module = _load_module(
    "audio_cpp_asr_adapter_test_module", "engines/adapters/asr_audio_cpp_adapter.py"
)
vc_module = _load_module(
    "audio_cpp_vc_adapter_test_module", "engines/adapters/audio_cpp_vc_adapter.py"
)
node_module = _load_module(
    "audio_cpp_multitask_node_test_module", "nodes/engines/audio_cpp_engine_node.py"
)


def _fake_save_factory(tmp_path):
    paths = []

    def save(_waveform, _sample_rate):
        path = tmp_path / f"audio-{len(paths)}.wav"
        path.write_bytes(b"wav")
        paths.append(path)
        return str(path)

    return paths, save


@pytest.mark.unit
def test_engine_node_advertises_asr_and_vc_consumers():
    asr_engine = node_module.AudioCppEngineNode().create_engine_config(
        "managed", "qwen3_asr", "auto", "auto", "cpu", 0, 4, "auto"
    )[0]
    vc_engine = node_module.AudioCppEngineNode().create_engine_config(
        "managed", "seed_vc", "auto", "auto", "cpu", 0, 4, "auto"
    )[0]

    assert asr_engine["config"]["task"] == "asr"
    assert asr_engine["capabilities"] == ["asr"]
    assert vc_engine["config"]["task"] == "vc"
    assert vc_engine["capabilities"] == ["voice_conversion"]


@pytest.mark.unit
def test_asr_adapter_normalizes_words_and_speaker_turns(monkeypatch, tmp_path):
    paths, fake_save = _fake_save_factory(tmp_path)
    monkeypatch.setattr(
        asr_module.AudioProcessingUtils, "save_audio_to_temp_file", staticmethod(fake_save)
    )

    class FakeSession:
        task = "asr"
        model_id = "vibe-asr"

        def run(self, request):
            assert Path(request["audio"]).is_absolute()
            assert Path(request["audio"]).is_file()
            return SimpleNamespace(raw={
                "text": "hello world",
                "language": "en",
                "words": [
                    {"word": "hello", "start_sample": 0, "end_sample": 8000},
                    {"word": "world", "start_sample": 8000, "end_sample": 16000},
                ],
                "speaker_turns": [
                    {
                        "start_sample": 0,
                        "end_sample": 16000,
                        "speaker_id": "Speaker 1",
                        "text": "hello world",
                    }
                ],
            })

    monkeypatch.setattr(session_module, "get_audio_cpp_session", lambda _config: FakeSession())
    adapter = asr_module.AudioCppASREngineAdapter({
        "engine_type": "audio_cpp",
        "config": {"family": "vibevoice_asr", "connection_mode": "external_server"},
    })
    result = adapter.transcribe(ASRRequest(
        audio={"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
        timestamps="word",
        diarization=True,
        chunk_size=0,
    ))

    assert result.text == "[Speaker 1] hello world"
    assert result.language == "en"
    assert result.segments[0].speaker == "Speaker 1"
    assert [word.text for word in result.segments[0].words] == ["hello", "world"]
    assert paths and not paths[0].exists()


@pytest.mark.unit
def test_asr_adapter_uses_suite_chunking_and_deduplicates_overlap(monkeypatch, tmp_path):
    paths, fake_save = _fake_save_factory(tmp_path)
    monkeypatch.setattr(
        asr_module.AudioProcessingUtils, "save_audio_to_temp_file", staticmethod(fake_save)
    )

    class FakeSession:
        task = "asr"
        model_id = "nemotron-asr"
        owned = True

        def __init__(self):
            self.requests = []
            self.restarts = 0
            self.texts = iter((
                "one two three",
                "three four five",
                "five six seven",
            ))

        def restart_owned_runtime(self):
            self.restarts += 1

        def run(self, request):
            self.requests.append(request)
            assert Path(request["audio"]).is_file()
            return SimpleNamespace(raw={"text": next(self.texts), "language": "en"})

    fake_session = FakeSession()
    monkeypatch.setattr(
        session_module, "get_audio_cpp_session", lambda _config: fake_session
    )
    adapter = asr_module.AudioCppASREngineAdapter({
        "engine_type": "audio_cpp",
        "config": {"family": "nemotron_asr", "connection_mode": "external_server"},
    })
    result = adapter.transcribe(ASRRequest(
        audio={"waveform": torch.zeros(1, 1, 80), "sample_rate": 10},
        chunk_size=4,
        overlap=2,
    ))

    assert result.text == "one two three four five six seven"
    assert len(fake_session.requests) == 3
    assert fake_session.restarts == 2
    assert result.raw["timing"]["suite_chunks"] == 3
    assert any("Suite-side ASR chunking" in note for note in result.raw["notes"])
    assert [chunk["text"] for chunk in result.raw["chunks"]] == [
        "one two three",
        "three four five",
        "five six seven",
    ]
    assert [(chunk["start"], chunk["end"]) for chunk in result.raw["chunks"]] == [
        (0.0, 4.0),
        (2.0, 6.0),
        (4.0, 8.0),
    ]
    assert paths and all(not path.exists() for path in paths)


@pytest.mark.unit
def test_vibevoice_diarization_keeps_native_chunking(monkeypatch, tmp_path):
    paths, fake_save = _fake_save_factory(tmp_path)
    monkeypatch.setattr(
        asr_module.AudioProcessingUtils, "save_audio_to_temp_file", staticmethod(fake_save)
    )
    captured = {}

    class FakeSession:
        task = "asr"
        model_id = "vibe-asr"

        def run(self, request):
            captured.update(request)
            return SimpleNamespace(raw={
                "text": "hello",
                "speaker_turns": [{
                    "start_sample": 0,
                    "end_sample": 16000,
                    "speaker_id": "1",
                    "text": "hello",
                }],
            })

    monkeypatch.setattr(
        session_module, "get_audio_cpp_session", lambda _config: FakeSession()
    )
    adapter = asr_module.AudioCppASREngineAdapter({
        "engine_type": "audio_cpp",
        "config": {"family": "vibevoice_asr", "connection_mode": "external_server"},
    })
    result = adapter.transcribe(ASRRequest(
        audio={"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000},
        diarization=True,
        chunk_size=30,
        overlap=2,
    ))

    assert captured["options"]["audio_chunk_mode"] == "fixed"
    assert captured["options"]["audio_chunk_seconds"] == 30
    assert result.text == "[Speaker 1] hello"
    assert len(paths) == 1 and not paths[0].exists()


@pytest.mark.unit
def test_vc_adapter_forces_vc_task_and_uses_source_target_audio(monkeypatch, tmp_path):
    paths, fake_save = _fake_save_factory(tmp_path)
    monkeypatch.setattr(
        vc_module.AudioProcessingUtils, "save_audio_to_temp_file", staticmethod(fake_save)
    )
    captured = {}

    class FakeSession:
        task = "vc"
        model_id = "seed-vc"
        family = "seed_vc"

        def run(self, request):
            captured.update(request)
            assert Path(request["audio"]).is_file()
            assert Path(request["voice_ref"]).is_file()
            return SimpleNamespace(
                waveform=torch.ones(1, 2400),
                sample_rate=24000,
                named_audio={},
            )

    def fake_session(config):
        assert config["requested_task"] == "vc"
        assert config["task"] == "vc"
        return FakeSession()

    monkeypatch.setattr(session_module, "get_audio_cpp_session", fake_session)
    adapter = vc_module.AudioCppVoiceConversionAdapter({
        "family": "seed_vc",
        "connection_mode": "managed",
    })
    audio = {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 16000}
    converted, info = adapter.convert_voice(audio, audio)

    assert converted["waveform"].shape == (1, 1, 2400)
    assert converted["sample_rate"] == 24000
    assert captured["source_audio"] == captured["audio"]
    assert captured["target_voice"] == captured["voice_ref"]
    assert "Seed-VC" not in info or "seed_vc" in info
    assert paths and all(not path.exists() for path in paths)
