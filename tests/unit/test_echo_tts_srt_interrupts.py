"""
Unit tests for Echo-TTS SRT interrupt handling.
Tests nodes/echo_tts/echo_tts_srt_processor.py without requiring ComfyUI server.
"""

import pytest
import sys
import os
import importlib.util
from pathlib import Path
from types import SimpleNamespace

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Set up minimal environment to avoid ComfyUI imports
os.environ.setdefault('COMFYUI_TESTING', '1')

# Skip these tests when local unit-test env does not include audio deps.
pytest.importorskip("torch")
pytest.importorskip("torchaudio")

# Load module directly to avoid package import side effects
processor_path = custom_node_root / "nodes" / "echo_tts" / "echo_tts_srt_processor.py"
spec = importlib.util.spec_from_file_location("echo_tts_srt_processor_module", processor_path)
echo_tts_srt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(echo_tts_srt_module)

EchoTTSSRTProcessor = echo_tts_srt_module.EchoTTSSRTProcessor


@pytest.mark.unit
class TestEchoTTSSRTInterruptHandling:
    """Interrupt handling behavior for Echo-TTS SRT processor."""

    def test_check_interrupt_raises_with_subtitle_and_character_context(self, monkeypatch):
        """_check_interrupt should include subtitle and character context in raised error."""
        processor = EchoTTSSRTProcessor.__new__(EchoTTSSRTProcessor)
        monkeypatch.setattr(
            echo_tts_srt_module.model_management,
            "interrupt_processing",
            True,
            raising=False,
        )

        with pytest.raises(InterruptedError, match="subtitle 2/5, character 'narrator'"):
            processor._check_interrupt(subtitle_index=1, total_subtitles=5, character="narrator")

    def test_process_srt_content_checks_interrupt_before_parsing(self):
        """process_srt_content should interrupt before invoking SRT parsing."""
        parser_called = {"called": False}

        class FakeSRTParser:
            def parse_srt_content(self, *_args, **_kwargs):
                parser_called["called"] = True
                return []

        processor = EchoTTSSRTProcessor.__new__(EchoTTSSRTProcessor)
        processor.srt_available = True
        processor.SRTParser = lambda: FakeSRTParser()
        processor.config = {}
        processor._check_interrupt = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InterruptedError("entry interrupt")
        )

        with pytest.raises(InterruptedError, match="entry interrupt"):
            processor.process_srt_content(
                srt_content="1\n00:00:00,000 --> 00:00:01,000\nHello\n",
                voice_mapping={},
                seed=1,
                timing_mode="concatenate",
                timing_params={},
            )

        assert parser_called["called"] is False

    def test_process_all_subtitles_checks_interrupt_in_subtitle_loop(self, monkeypatch):
        """_process_all_subtitles should check interrupt at the top of each subtitle iteration."""
        processor = EchoTTSSRTProcessor.__new__(EchoTTSSRTProcessor)
        processor.config = {}

        subtitle = SimpleNamespace(
            text="Hello world",
            sequence=1,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
        )

        monkeypatch.setattr(
            echo_tts_srt_module.character_parser,
            "split_by_character",
            lambda _text, include_language=False: [],
            raising=True,
        )

        interrupt_calls = []

        def interrupt_stub(subtitle_index=None, total_subtitles=None, character=None):
            interrupt_calls.append((subtitle_index, total_subtitles, character))
            if subtitle_index == 0 and character is None:
                raise InterruptedError("subtitle interrupt")

        processor._check_interrupt = interrupt_stub

        with pytest.raises(InterruptedError, match="subtitle interrupt"):
            processor._process_all_subtitles([subtitle], voice_mapping={}, seed=42)

        assert interrupt_calls[0] == (0, 1, None)

    def test_process_all_subtitles_checks_interrupt_in_tts_callback(self, monkeypatch):
        """_process_all_subtitles should check interrupt again immediately before segment TTS generation."""
        processor = EchoTTSSRTProcessor.__new__(EchoTTSSRTProcessor)
        processor.config = {}

        class FakeAdapter:
            def update_config(self, _config):
                return None

            def process_text(self, **_kwargs):
                return echo_tts_srt_module.torch.zeros(1, 64)

        processor.processor = FakeAdapter()

        subtitle = SimpleNamespace(
            text="Hello world",
            sequence=1,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
        )

        segment = SimpleNamespace(character="narrator", text="Hello world", parameters={})

        monkeypatch.setattr(
            echo_tts_srt_module.character_parser,
            "split_by_character",
            lambda _text, include_language=False: [],
            raising=True,
        )
        monkeypatch.setattr(
            echo_tts_srt_module.character_parser,
            "parse_text_segments",
            lambda _text: [segment],
            raising=True,
        )

        character_checks = {"count": 0}

        def interrupt_stub(subtitle_index=None, total_subtitles=None, character=None):
            if character is not None:
                character_checks["count"] += 1
                # First check is at segment loop entry, second check is in _tts_generate_func.
                if character_checks["count"] == 2:
                    raise InterruptedError("callback interrupt")

        processor._check_interrupt = interrupt_stub

        with pytest.raises(InterruptedError, match="callback interrupt"):
            processor._process_all_subtitles([subtitle], voice_mapping={"narrator": {}}, seed=7)

        assert character_checks["count"] == 2
