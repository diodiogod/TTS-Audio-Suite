"""Focused tests for the audio.cpp adapter, processors, and engine node."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter_module = _load_module("audio_cpp_adapter_test_module", "engines/adapters/audio_cpp_adapter.py")
processor_module = _load_module("audio_cpp_processor_test_module", "nodes/audio_cpp/audio_cpp_processor.py")
srt_module = _load_module("audio_cpp_srt_test_module", "nodes/audio_cpp/audio_cpp_srt_processor.py")
node_module = _load_module("audio_cpp_engine_node_test_module", "nodes/engines/audio_cpp_engine_node.py")


class _FakeSession:
    def __init__(self, sample_rate=32000):
        self.sample_rate = sample_rate
        self.requests = []
        self.owned = True
        self.endpoint = ""
        self.model_id = "owned-test-model"
        self.family = "qwen3_tts"
        self.config = {"task": "tts"}

    def run(self, request):
        self.requests.append(dict(request))
        # Owned sessions receive a random HTTP endpoint only after the server
        # starts. That transient port must not change the audio cache identity.
        self.endpoint = "http://127.0.0.1:54321"
        if request.get("voice_ref"):
            from pathlib import Path

            assert Path(request["voice_ref"]).is_absolute()
            assert Path(request["voice_ref"]).is_file()
        return SimpleNamespace(
            waveform=torch.ones(1, self.sample_rate // 10),
            sample_rate=self.sample_rate,
            named_audio={},
        )


def test_adapter_caches_real_sample_rate_and_cleans_reference(monkeypatch, tmp_path):
    adapter_module.get_audio_cache().clear_cache()
    adapter_module._CACHE_SAMPLE_RATES.clear()
    session = _FakeSession(sample_rate=32000)
    monkeypatch.setattr(adapter_module, "_get_session", lambda config: session)
    created = []

    def fake_save(waveform, sample_rate):
        path = tmp_path / f"reference-{len(created)}.wav"
        path.write_bytes(b"temporary")
        created.append(path)
        return str(path)

    monkeypatch.setattr(
        adapter_module.AudioProcessingUtils, "save_audio_to_temp_file", staticmethod(fake_save)
    )
    adapter = adapter_module.AudioCppEngineAdapter(
        {"family": "qwen3_tts", "package_id": "qwen3_tts_1_7b_base_q8_0", "task": "tts"}
    )
    voice = {"audio": {"waveform": torch.zeros(1, 80), "sample_rate": 16000}}

    first, first_rate = adapter.generate_single("hello", voice, seed=7)
    second, second_rate = adapter.generate_single("hello", voice, seed=7)

    assert first_rate == second_rate == 32000
    assert torch.equal(first, second)
    assert len(session.requests) == 1
    assert session.requests[0]["seed"] == "7"
    assert len(created) == 1
    assert created[0].exists()
    adapter.close()
    assert all(not path.exists() for path in created)


class _ProcessorAdapter:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.config = {}

    def update_config(self, config):
        self.config = dict(config)

    def generate_single(self, **kwargs):
        return torch.ones(1, self.sample_rate // 10), self.sample_rate


def test_processor_materializes_leading_pause_at_response_rate(monkeypatch):
    segment = SimpleNamespace(
        text="[pause:0.01] hello",
        character="narrator",
        parameters={},
        language=None,
        explicit_language=False,
    )
    monkeypatch.setattr(processor_module.AudioCppProcessor, "_setup_character_parser", lambda self, text: None)
    monkeypatch.setattr(
        processor_module.character_parser,
        "parse_text_segments",
        lambda text, engine_type=None: [segment],
    )
    monkeypatch.setattr(processor_module, "get_character_mapping", lambda *args, **kwargs: {})
    processor = processor_module.AudioCppProcessor(_ProcessorAdapter(32000), {"language": "auto"})

    records = processor.process_text(
        "ignored", {}, seed=1, enable_chunking=False, show_text_logging=False
    )

    assert processor.sample_rate == 32000
    assert records[0]["sample_rate"] == 32000
    assert records[0]["waveform"].shape == (1, 320)
    assert records[1]["sample_rate"] == 32000


def test_processor_uses_glm_transcripts_and_resets_rate_between_generations(monkeypatch):
    segment = SimpleNamespace(
        text="hello",
        character="Alice",
        parameters={},
        language=None,
        explicit_language=False,
    )
    monkeypatch.setattr(
        processor_module.AudioCppProcessor, "_setup_character_parser", lambda self, text: None
    )
    monkeypatch.setattr(
        processor_module.character_parser,
        "parse_text_segments",
        lambda text, engine_type=None: [segment],
    )
    discovery_modes = []

    def mapping(characters, engine_type):
        discovery_modes.append(engine_type)
        return {"Alice": (None, None)}

    monkeypatch.setattr(processor_module, "get_character_mapping", mapping)
    adapter = _ProcessorAdapter(24000)
    processor = processor_module.AudioCppProcessor(adapter, {"family": "glm_tts"})

    processor.process_text("first", {}, seed=1, enable_chunking=False, show_text_logging=False)
    adapter.sample_rate = 32000
    processor.process_text("second", {}, seed=1, enable_chunking=False, show_text_logging=False)

    assert discovery_modes == ["audio_and_text", "audio_and_text"]
    assert processor.sample_rate == 32000


def test_processor_rejects_mixed_response_rates():
    processor = processor_module.AudioCppProcessor(_ProcessorAdapter(), {})
    segments = [
        {"waveform": torch.zeros(1, 8), "sample_rate": 24000, "text": "a"},
        {"waveform": torch.zeros(1, 8), "sample_rate": 32000, "text": "b"},
    ]
    with pytest.raises(RuntimeError, match="inconsistent sample rates"):
        processor.combine_audio_segments(segments)


def test_engine_node_resolves_owned_package_task(monkeypatch):
    monkeypatch.setattr(node_module, "_recommended_package", lambda family: "design-package")
    monkeypatch.setattr(node_module, "_validate_package", lambda family, package: None)
    monkeypatch.setattr(node_module, "_resolve_task", lambda family, package, task: "vdes")

    engine = node_module.AudioCppEngineNode().create_engine_config(
        "managed", "qwen3_tts", "auto", "auto", "cuda", 0, 4, "auto"
    )[0]

    assert engine["config"]["package_id"] == "design-package"
    assert engine["config"]["task"] == "vdes"
    assert engine["config"]["threads"] == 4
    assert engine["capabilities"] == ["tts", "voice_design"]


def test_engine_node_keeps_external_server_task_authoritative():
    engine = node_module.AudioCppEngineNode().create_engine_config(
        "external_server",
        "qwen3_tts",
        "auto",
        "auto",
        "cpu",
        0,
        4,
        "auto",
        server_url="http://127.0.0.1:8080",
    )[0]
    assert engine["config"]["task"] == "auto"
    assert engine["config"]["package_id"] == "auto"


def test_engine_node_uses_pinned_catalog_contract():
    inputs = node_module.AudioCppEngineNode.INPUT_TYPES()
    assert "qwen3_tts" in inputs["required"]["family"][0]
    assert "qwen3_tts_1_7b_base_q8_0" in inputs["required"]["package_id"][0]

    engine = node_module.AudioCppEngineNode().create_engine_config(
        "managed", "qwen3_tts", "auto", "auto", "cpu", 0, 4, "auto"
    )[0]
    assert engine["config"]["package_id"] == "qwen3_tts_1_7b_base_q8_0"
    assert engine["config"]["task"] == "tts"


def test_unified_nodes_construct_audio_cpp_processors_without_nodes_package_collision():
    text_module = _load_module(
        "audio_cpp_unified_text_test_module", "nodes/unified/tts_text_node.py"
    )
    srt_unified_module = _load_module(
        "audio_cpp_unified_srt_test_module", "nodes/unified/tts_srt_node.py"
    )
    engine = node_module.AudioCppEngineNode().create_engine_config(
        "external_server",
        "pocket_tts",
        "auto",
        "auto",
        "cpu",
        0,
        4,
        "auto",
        server_url="http://127.0.0.1:9999",
        model_id="wiring-only",
    )[0]

    text_wrapper = text_module.UnifiedTTSTextNode()._create_proper_engine_node_instance(engine)
    srt_wrapper = srt_unified_module.UnifiedTTSSRTNode()._create_proper_engine_node_instance(engine)

    assert type(text_wrapper.adapter).__name__ == "AudioCppEngineAdapter"
    assert type(text_wrapper.processor).__name__ == "AudioCppProcessor"
    assert type(srt_wrapper.processor).__name__ == "AudioCppSRTProcessor"


def test_unified_nodes_surface_audio_cpp_runtime_errors(monkeypatch):
    from utils.audio_cpp import session as session_module

    text_module = _load_module(
        "audio_cpp_unified_text_error_test_module", "nodes/unified/tts_text_node.py"
    )
    srt_unified_module = _load_module(
        "audio_cpp_unified_srt_error_test_module", "nodes/unified/tts_srt_node.py"
    )
    engine = node_module.AudioCppEngineNode().create_engine_config(
        "external_server",
        "pocket_tts",
        "auto",
        "auto",
        "cpu",
        0,
        4,
        "auto",
        server_url="http://127.0.0.1:9999",
        model_id="error-only",
    )[0]

    def fail_session(config):
        raise RuntimeError("visible audio.cpp failure")

    monkeypatch.setattr(session_module, "get_audio_cpp_session", fail_session)

    with pytest.raises(RuntimeError, match="visible audio.cpp failure"):
        text_module.UnifiedTTSTextNode().generate_speech(
            engine, "hello", "none", 1, enable_chunking=False, enable_audio_cache=False
        )
    with pytest.raises(RuntimeError, match="visible audio.cpp failure"):
        srt_unified_module.UnifiedTTSSRTNode().generate_srt_speech(
            engine,
            "1\n00:00:00,000 --> 00:00:01,000\nhello",
            "none",
            1,
            "concatenate",
            enable_audio_cache=False,
        )


class _Subtitle:
    def __init__(self, sequence, text, start, end):
        self.sequence = sequence
        self.text = text
        self.start_time = start
        self.end_time = end
        self.duration = end - start


class _SRTTextProcessor:
    sample_rate = 16000

    def reset_sample_rate(self):
        return None

    def process_text(self, **kwargs):
        return [{"waveform": torch.ones(1, 8000), "sample_rate": 16000, "text": kwargs["text"]}]

    def combine_audio_segments(self, records, **kwargs):
        return records[0]["waveform"]


def test_srt_delays_blank_cue_until_dynamic_rate_is_known(monkeypatch):
    subtitles = [_Subtitle(1, "", 0.0, 0.25), _Subtitle(2, "hello", 0.25, 0.75)]
    instance = srt_module.AudioCppSRTProcessor.__new__(srt_module.AudioCppSRTProcessor)
    instance.config = {}
    instance._processor = _SRTTextProcessor()
    instance.SRTParser = lambda: SimpleNamespace(
        parse_srt_content=lambda content, allow_overlaps: subtitles
    )
    monkeypatch.setattr(instance, "_check_interrupt", lambda *args: None)
    monkeypatch.setattr(srt_module.SRTOverlapHandler, "detect_overlaps", lambda items: False)
    monkeypatch.setattr(
        srt_module.SRTOverlapHandler,
        "handle_smart_natural_fallback",
        lambda mode, overlaps, label: (mode, False),
    )
    captured = {}

    def fake_assemble(audio, subs, mode, params, rate):
        captured["segments"] = audio
        return torch.cat(audio, dim=-1), None, None

    monkeypatch.setattr(instance, "_assemble", fake_assemble)
    monkeypatch.setattr(
        srt_module,
        "SRTReportGenerator",
        lambda: SimpleNamespace(
            generate_timing_report=lambda *args: "report",
            generate_adjusted_srt_string=lambda *args: "adjusted",
        ),
    )

    audio, _, report, adjusted = instance.process_srt_content(
        "unused", {}, 0, "concatenate", {}, enable_audio_cache=False
    )

    assert captured["segments"][0].shape[-1] == 4000
    assert audio["sample_rate"] == 16000
    assert audio["waveform"].shape == (1, 1, 12000)
    assert (report, adjusted) == ("report", "adjusted")
