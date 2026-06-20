import pytest
from engines.adapters.asr_granite_adapter import parse_granite_native_timestamps
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
