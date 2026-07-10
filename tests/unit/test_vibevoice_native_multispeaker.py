import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
ADAPTER_PATH = REPO_ROOT / "engines" / "adapters" / "vibevoice_adapter.py"
SPEC = importlib.util.spec_from_file_location("vibevoice_adapter_test_module", ADAPTER_PATH)
ADAPTER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER_MODULE)


class FakeVibeVoiceEngine:
    def __init__(self):
        self.calls = []

    def _prepare_voice_samples(self, speaker_voices):
        return speaker_voices

    def generate_speech(self, **kwargs):
        self.calls.append(kwargs)
        return {"waveform": None}


@pytest.mark.unit
def test_native_multispeaker_remaps_global_slots_to_local_sequence(monkeypatch):
    fake_engine = FakeVibeVoiceEngine()

    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "vibevoice_engine",
        property(lambda self: fake_engine),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_model",
        property(lambda self: object()),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_processor",
        property(lambda self: object()),
    )

    adapter = ADAPTER_MODULE.VibeVoiceEngineAdapter(SimpleNamespace(), {})

    narrator_voice = {"audio_path": "narrator.wav"}
    bob_alias_voice = {"audio_path": "bob_alias.wav"}
    speaker3_override = {"audio_path": "speaker3.wav"}

    adapter._generate_native_multispeaker(
        segments=[("Bob", "Hi there."), ("narrator", "Welcome back.")],
        voice_mapping={"narrator": narrator_voice, "Bob": bob_alias_voice},
        params={"speaker3_voice": speaker3_override, "enable_cache": False},
        global_char_to_speaker={"narrator": 1, "Alice": 2, "Bob": 3},
    )

    call = fake_engine.calls[0]
    assert call["text"] == "Speaker 1: Hi there.\nSpeaker 2: Welcome back."
    assert call["voice_samples"] == [speaker3_override, narrator_voice]


@pytest.mark.unit
def test_native_multispeaker_does_not_fake_speaker1_from_character_voice(monkeypatch):
    fake_engine = FakeVibeVoiceEngine()

    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "vibevoice_engine",
        property(lambda self: fake_engine),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_model",
        property(lambda self: object()),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_processor",
        property(lambda self: object()),
    )

    adapter = ADAPTER_MODULE.VibeVoiceEngineAdapter(SimpleNamespace(), {})

    alice_voice = {"audio_path": "alice.wav"}
    clint_voice = {"audio_path": "clint.wav"}

    adapter._generate_native_multispeaker(
        segments=[("female_01", "Hi there."), ("clint_eastwood cc3 (enhanced2)", "Hello there.")],
        voice_mapping={
            "female_01": alice_voice,
            "clint_eastwood cc3 (enhanced2)": clint_voice,
        },
        params={"enable_cache": False},
        global_char_to_speaker={
            "clint_eastwood cc3 (enhanced2)": 1,
            "female_01": 2,
            "male_01": 3,
        },
    )

    call = fake_engine.calls[0]
    assert call["text"] == "Speaker 1: Hi there.\nSpeaker 2: Hello there."
    assert call["voice_samples"] == [alice_voice, clint_voice]


@pytest.mark.unit
def test_native_multispeaker_narrator_input_does_not_override_tagged_global_speaker1(monkeypatch):
    fake_engine = FakeVibeVoiceEngine()

    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "vibevoice_engine",
        property(lambda self: fake_engine),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_model",
        property(lambda self: object()),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_processor",
        property(lambda self: object()),
    )

    adapter = ADAPTER_MODULE.VibeVoiceEngineAdapter(SimpleNamespace(), {})

    narrator_voice = {"audio_path": "joe.wav"}
    bob_voice = {"audio_path": "bob.wav"}
    alice_voice = {"audio_path": "alice.wav"}

    adapter._generate_native_multispeaker(
        segments=[("male_01", "Hello."), ("female_01", "Hi there.")],
        voice_mapping={
            "narrator": narrator_voice,
            "male_01": bob_voice,
            "female_01": alice_voice,
        },
        params={"enable_cache": False},
        global_char_to_speaker={
            "narrator": 1,
            "female_01": 3,
            "male_01": 4,
        },
    )

    call = fake_engine.calls[0]
    assert call["text"] == "Speaker 1: Hello.\nSpeaker 2: Hi there."
    assert call["voice_samples"] == [bob_voice, alice_voice]


@pytest.mark.unit
def test_native_multispeaker_narrator_input_overrides_first_character_when_no_narrator_exists(monkeypatch):
    fake_engine = FakeVibeVoiceEngine()

    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "vibevoice_engine",
        property(lambda self: fake_engine),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_model",
        property(lambda self: object()),
    )
    monkeypatch.setattr(
        ADAPTER_MODULE.VibeVoiceEngineAdapter,
        "current_processor",
        property(lambda self: object()),
    )

    adapter = ADAPTER_MODULE.VibeVoiceEngineAdapter(SimpleNamespace(), {})

    narrator_voice = {"audio_path": "joe.wav"}
    bob_voice = {"audio_path": "bob.wav"}
    alice_voice = {"audio_path": "alice.wav"}

    adapter._generate_native_multispeaker(
        segments=[("male_01", "Hello."), ("female_01", "Hi there.")],
        voice_mapping={
            "narrator": narrator_voice,
            "male_01": bob_voice,
            "female_01": alice_voice,
        },
        params={"enable_cache": False},
        global_char_to_speaker={
            "male_01": 1,
            "female_01": 2,
        },
    )

    call = fake_engine.calls[0]
    assert call["text"] == "Speaker 1: Hello.\nSpeaker 2: Hi there."
    assert call["voice_samples"] == [narrator_voice, alice_voice]
