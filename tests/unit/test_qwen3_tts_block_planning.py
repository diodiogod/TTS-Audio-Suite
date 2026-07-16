import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROCESSOR_PATH = REPO_ROOT / "nodes" / "qwen3_tts" / "qwen3_tts_processor.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen3_tts_processor_test_module", PROCESSOR_PATH
)
PROCESSOR_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESSOR_MODULE)
Qwen3TTSProcessor = PROCESSOR_MODULE.Qwen3TTSProcessor


class FakeChunker:
    def split_into_chunks(self, text, max_chars):
        return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]


@pytest.mark.unit
def test_block_planning_counts_pause_delimited_generation_calls():
    processor = object.__new__(Qwen3TTSProcessor)
    processor.chunker = FakeChunker()
    segments = [
        SimpleNamespace(text="First line.[pause:1]Second line."),
        SimpleNamespace(text="Third line."),
        SimpleNamespace(text="Fourth line."),
    ]

    lengths = processor._plan_generation_block_lengths(segments, False, 1000)

    assert lengths == [11, 12, 11, 12]


@pytest.mark.unit
def test_block_planning_counts_chunks_inside_pause_sections():
    processor = object.__new__(Qwen3TTSProcessor)
    processor.chunker = FakeChunker()
    segments = [SimpleNamespace(text="abcdefgh[pause:0.5]ijklmnop")]

    lengths = processor._plan_generation_block_lengths(segments, True, 4)

    assert lengths == [4, 4, 4, 4]
