import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / "utils" / "runtimes" / "workers" / "fish_audio_s2_worker.py"
SPEC = importlib.util.spec_from_file_location("fish_audio_s2_worker_test_module", WORKER_PATH)
WORKER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER_MODULE)


class FakeReferenceLoader:
    def __init__(self):
        self.ref_by_hash = {}
        self.encoded = []

    def encode_reference(self, reference_audio, enable_reference_audio):
        assert enable_reference_audio is True
        token = f"encoded-{len(self.encoded)}"
        self.encoded.append(reference_audio)
        return token


@pytest.mark.unit
def test_reference_cache_reuses_audio_but_not_previous_speaker_tag():
    loader = FakeReferenceLoader()

    first_tokens, first_texts = WORKER_MODULE._load_references_preserving_current_speaker_tags(
        loader,
        [SimpleNamespace(audio=b"joe", text="<|speaker:0|>Joe reference")],
        "on",
    )
    second_tokens, second_texts = WORKER_MODULE._load_references_preserving_current_speaker_tags(
        loader,
        [
            SimpleNamespace(audio=b"bob", text="<|speaker:0|>Bob reference"),
            SimpleNamespace(audio=b"joe", text="<|speaker:1|>Joe reference"),
        ],
        "on",
    )

    assert first_texts == ["<|speaker:0|>Joe reference"]
    assert second_tokens == ["encoded-1", first_tokens[0]]
    assert second_texts == ["<|speaker:0|>Bob reference", "<|speaker:1|>Joe reference"]
