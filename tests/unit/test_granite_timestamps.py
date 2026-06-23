import pytest
import torch
from engines.adapters.asr_granite_adapter import (
    format_granite_diarization_segments,
    parse_granite_native_timestamps,
    parse_granite_diarization,
    remap_words_to_speaker_segments,
)
from utils.asr.types import ASRSegment, ASRWord


def test_parse_simple_timestamps():
    # Arrange
    ts_text = "hello [T:45] world [T:82]"
    chunk_offset = 0.0

    # Act
    plain_text, segments = parse_granite_native_timestamps(ts_text, chunk_offset)

    # Assert
    assert plain_text == "hello world"
    assert len(segments) == 2

    # First word
    assert segments[0].text == "hello"
    assert pytest.approx(segments[0].start) == 0.0
    assert pytest.approx(segments[0].end) == 0.45
    assert len(segments[0].words) == 1
    assert segments[0].words[0].text == "hello"
    assert pytest.approx(segments[0].words[0].start) == 0.0
    assert pytest.approx(segments[0].words[0].end) == 0.45

    # Second word
    assert segments[1].text == "world"
    assert pytest.approx(segments[1].start) == 0.45
    assert pytest.approx(segments[1].end) == 0.82
    assert len(segments[1].words) == 1
    assert segments[1].words[0].text == "world"
    assert pytest.approx(segments[1].words[0].start) == 0.45
    assert pytest.approx(segments[1].words[0].end) == 0.82


def test_parse_timestamps_with_rollover():
    # Arrange
    # Tag values: 990 (9.9s), 20 (rolls over to 10.2s), 120 (rolls over to 11.2s)
    ts_text = "hello [T:990] world [T:20] next [T:120]"
    chunk_offset = 0.0

    # Act
    plain_text, segments = parse_granite_native_timestamps(ts_text, chunk_offset)

    # Assert
    assert plain_text == "hello world next"
    assert len(segments) == 3

    # first: 0.0s -> 9.90s
    assert segments[0].text == "hello"
    assert pytest.approx(segments[0].start) == 0.0
    assert pytest.approx(segments[0].end) == 9.90

    # second: 9.90s -> 10.20s
    assert segments[1].text == "world"
    assert pytest.approx(segments[1].start) == 9.90
    assert pytest.approx(segments[1].end) == 10.20

    # third: 10.20s -> 11.20s
    assert segments[2].text == "next"
    assert pytest.approx(segments[2].start) == 10.20
    assert pytest.approx(segments[2].end) == 11.20


def test_parse_timestamps_with_silence():
    # Arrange
    # silence '_' ends at 1.0s, 'hello' ends at 2.0s
    ts_text = "_ [T:100] hello [T:200]"
    chunk_offset = 5.0  # test offset shift

    # Act
    plain_text, segments = parse_granite_native_timestamps(ts_text, chunk_offset)

    # Assert
    assert plain_text == "hello"
    assert len(segments) == 1  # silence '_' should be filtered out

    # hello starts at 1.0s (+5.0s offset = 6.0s) and ends at 2.0s (+5.0s offset = 7.0s)
    assert segments[0].text == "hello"
    assert pytest.approx(segments[0].start) == 6.0
    assert pytest.approx(segments[0].end) == 7.0


def test_parse_empty_and_malformed():
    assert parse_granite_native_timestamps("", 0.0) == ("", [])
    assert parse_granite_native_timestamps("   ", 10.0) == ("", [])
    assert parse_granite_native_timestamps("hello [T:invalid] world [T:50]", 0.0) == ("world", [
        # world ends at 0.5s. But what is its start time?
        # Since 'hello [T:invalid]' was skipped, the start time defaults to 0.0.
        ASRSegment(
            start=0.0,
            end=0.5,
            text="world",
            words=[ASRWord(start=0.0, end=0.5, text="world")],
        )
    ])


def test_parse_granite_diarization():
    # Arrange
    text = "[Speaker 1]: hello there. [Speaker 2]: hi how are you? [Speaker 1]: good."
    chunk_offset = 0.0
    chunk_duration = 5.0

    # Act
    plain_text, segments = parse_granite_diarization(text, chunk_offset, chunk_duration)

    # Assert
    # Math: hello there. (12 chars), hi how are you? (15 chars), good. (5 chars) -> total 32
    # chunk_duration = 5.0
    assert segments[0].speaker == "Speaker 1"
    assert segments[0].text == "hello there."
    assert abs(segments[0].start - 0.0) < 1e-5
    assert abs(segments[0].end - (5.0 * 12 / 32)) < 1e-5

    assert segments[1].speaker == "Speaker 2"
    assert segments[1].text == "hi how are you?"
    assert abs(segments[1].start - (5.0 * 12 / 32)) < 1e-5
    assert abs(segments[1].end - (5.0 * 27 / 32)) < 1e-5

    assert segments[2].speaker == "Speaker 1"
    assert segments[2].text == "good."
    assert abs(segments[2].start - (5.0 * 27 / 32)) < 1e-5
    assert abs(segments[2].end - 5.0) < 1e-5


def test_format_granite_diarization_segments_uses_suite_character_style():
    segments = [
        ASRSegment(start=0.0, end=1.0, text="hello there.", speaker="Speaker 1"),
        ASRSegment(start=1.0, end=2.0, text="hi.", speaker="Speaker 2"),
    ]

    assert format_granite_diarization_segments(segments) == "[Speaker 1] hello there. [Speaker 2] hi."


def test_remap_words_to_speaker_segments():
    # Arrange
    speaker_segments = [
        ASRSegment(start=0.0, end=5.0, text="hello there", speaker="Speaker 1"),
        ASRSegment(start=0.0, end=5.0, text="hi how are you", speaker="Speaker 2")
    ]
    aligned_words = [
        ASRWord(start=0.1, end=0.4, text="hello"),
        ASRWord(start=0.5, end=0.8, text="there"),
        ASRWord(start=1.2, end=1.4, text="hi"),
        ASRWord(start=1.5, end=1.7, text="how"),
        ASRWord(start=1.8, end=2.0, text="are"),
        ASRWord(start=2.1, end=2.5, text="you")
    ]

    # Act
    remap_words_to_speaker_segments(speaker_segments, aligned_words)

    # Assert
    assert len(speaker_segments[0].words) == 2
    assert speaker_segments[0].words[0].text == "hello"
    assert speaker_segments[0].words[1].text == "there"
    assert pytest.approx(speaker_segments[0].start) == 0.1
    assert pytest.approx(speaker_segments[0].end) == 0.8

    assert len(speaker_segments[1].words) == 4
    assert speaker_segments[1].words[0].text == "hi"
    assert speaker_segments[1].words[-1].text == "you"
    assert pytest.approx(speaker_segments[1].start) == 1.2
    assert pytest.approx(speaker_segments[1].end) == 2.5


def test_diarization_fallback_warning():
    from engines.adapters.asr_granite_adapter import GraniteASREngineAdapter
    from utils.asr.types import ASRRequest

    # Arrange
    engine_data = {
        "engine_type": "granite_asr",
        "config": {
            "model_name": "granite-4.0-1b-speech" # Non-plus model
        }
    }
    adapter = GraniteASREngineAdapter(engine_data)
    
    # Mock runtime and audio prep
    class MockRuntime:
        def transcribe(self, *args, **kwargs):
            return {"text": "hello", "language": "en"}
            
    adapter._get_model = lambda: MockRuntime()
    adapter._prepare_audio = lambda x: torch.zeros(16000)
    
    req = ASRRequest(
        audio={"waveform": torch.zeros(16000), "sample_rate": 16000},
        diarization=True, # Requested
        timestamps="none"
    )
    
    # Act
    result = adapter.transcribe(req)
    
    # Assert
    assert result.raw is not None
    assert "warnings" in result.raw
    assert any("Only 'plus' variants support speaker attribution natively" in w for w in result.raw["warnings"])


def test_diarization_qwen_realignment_transcribe():
    from engines.adapters.asr_granite_adapter import GraniteASREngineAdapter
    from utils.asr.types import ASRRequest

    # Arrange
    engine_data = {
        "engine_type": "granite_asr",
        "config": {
            "model_name": "granite-speech-4.1-2b-plus", # Plus model
            "asr_use_forced_aligner": True
        }
    }
    adapter = GraniteASREngineAdapter(engine_data)
    
    class MockRuntime:
        def transcribe(self, *args, **kwargs):
            # Return speaker attributed output
            return {"text": "[Speaker 1]: hello there. [Speaker 2]: hi.", "language": "en"}
            
    class MockAlignerItem:
        def __init__(self, text, start_time, end_time):
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class MockAligner:
        def align(self, audio, text, language):
            # Mock alignment output for "hello there hi"
            return [[
                MockAlignerItem("hello", 0.1, 0.4),
                MockAlignerItem("there", 0.5, 0.8),
                MockAlignerItem("hi", 1.2, 1.4)
            ]]

    adapter._get_model = lambda: MockRuntime()
    adapter._get_forced_aligner = lambda: MockAligner()
    adapter._prepare_audio = lambda x: torch.zeros(16000)
    adapter._can_use_aligner = lambda *args: True
    
    req = ASRRequest(
        audio={"waveform": torch.zeros(16000), "sample_rate": 16000},
        diarization=True,
        timestamps="word",
        chunk_size=0 # avoid chunking loop complexities
    )
    
    # Act
    result = adapter.transcribe(req)
    
    # Assert
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "Speaker 1"
    assert result.segments[0].text == "[Speaker 1] hello there."
    assert len(result.segments[0].words) == 2
    assert result.segments[0].words[0].text == "hello"
    assert result.segments[0].words[0].start == 0.1
    assert result.segments[0].words[0].end == 0.4
    assert result.segments[0].words[1].text == "there"
    assert result.segments[0].words[1].start == 0.5
    assert result.segments[0].words[1].end == 0.8
    assert result.segments[1].text == "[Speaker 2] hi."
    assert result.text == "[Speaker 1] hello there. [Speaker 2] hi."
    assert result.segments[0].start == 0.1
    assert result.segments[0].end == 0.8

    assert result.segments[1].speaker == "Speaker 2"
    assert result.segments[1].text == "[Speaker 2]: hi."
    assert len(result.segments[1].words) == 1
    assert result.segments[1].words[0].text == "hi"
    assert result.segments[1].words[0].start == 1.2
    assert result.segments[1].words[0].end == 1.4
    assert result.segments[1].start == 1.2
    assert result.segments[1].end == 1.4
