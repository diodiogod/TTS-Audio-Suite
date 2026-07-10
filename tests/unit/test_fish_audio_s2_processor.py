import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PROCESSOR_PATH = (
    REPO_ROOT
    / "nodes"
    / "fish_audio_s2"
    / "fish_audio_s2_processor.py"
)
SPEC = importlib.util.spec_from_file_location("fish_audio_s2_processor_test_module", PROCESSOR_PATH)
PROCESSOR_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESSOR_MODULE)


class FakeCharacterParser:
    def set_available_characters(self, _characters):
        pass

    def set_character_language_default(self, _character, _language):
        pass

    def reset_session_cache(self):
        pass

    def parse_text_segments(self, text, engine_type=None):
        characters = ["alice", "bob", "rick"] if text == "all" else ["bob", "rick"]
        return [
            SimpleNamespace(character=character, text=character.title(), parameters={}, language="en")
            for character in characters
        ]


class CapturingAdapter:
    def __init__(self):
        self.calls = []

    def update_config(self, _config):
        pass

    def generate_dialogue(self, turns, voice_refs, **_kwargs):
        self.calls.append((list(turns), list(voice_refs)))
        return torch.zeros(1, 16)


@pytest.mark.unit
def test_srt_uses_global_reference_lookup_with_local_speaker_ids(monkeypatch):
    parser = FakeCharacterParser()
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(PROCESSOR_MODULE, "get_character_mapping", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    references = [{"id": "bob"}, {"id": "rick"}]
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {"speaker_references": references},
    )

    reference_order = processor.get_character_order("all")
    processor.process_text("subtitle", {}, seed=1, reference_order=reference_order)

    turns, used_references = adapter.calls[0]
    assert turns == [(0, "Bob"), (1, "Rick")]
    assert [reference["id"] for reference in used_references] == ["bob", "rick"]


@pytest.mark.unit
def test_custom_switching_generates_each_segment_as_local_speaker_zero(monkeypatch):
    parser = FakeCharacterParser()
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(PROCESSOR_MODULE, "get_character_mapping", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    references = [{"id": "bob"}, {"id": "rick"}]
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": references,
            "multi_speaker_mode": "Custom Character Switching",
        },
    )

    reference_order = processor.get_character_order("all")
    records = processor.process_text(
        "subtitle", {}, seed=1, reference_order=reference_order
    )

    assert [call[0] for call in adapter.calls] == [[(0, "Bob")], [(0, "Rick")]]
    assert [[ref["id"] for ref in call[1]] for call in adapter.calls] == [["bob"], ["rick"]]
    assert [record["text"] for record in records] == ["Bob", "Rick"]


@pytest.mark.unit
def test_language_prompting_prepends_shared_display_name(monkeypatch):
    parser = FakeCharacterParser()

    def parse_with_language(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(character="alice", text="Hallo", parameters={}, language="de", explicit_language=False),
        ]

    parser.parse_text_segments = parse_with_language
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(PROCESSOR_MODULE, "get_character_mapping", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"id": "alice"}],
            "multi_speaker_mode": "Custom Character Switching",
            "language_prompting": "Auto Inline Tag",
        },
    )

    processor.process_text("subtitle", {}, seed=1, reference_order=["alice"], show_text_logging=False)

    turns, _used_references = adapter.calls[0]
    assert turns == [(0, "<German> Hallo")]


@pytest.mark.unit
def test_explicit_english_language_prompt_is_preserved(monkeypatch):
    parser = FakeCharacterParser()

    def parse_with_explicit_english(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(character="alice", text="Hello", parameters={}, language="en", explicit_language=True),
        ]

    parser.parse_text_segments = parse_with_explicit_english
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(PROCESSOR_MODULE, "get_character_mapping", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"id": "alice"}],
            "multi_speaker_mode": "Custom Character Switching",
            "language_prompting": "Auto Inline Tag",
        },
    )

    processor.process_text("subtitle", {}, seed=1, reference_order=["alice"], show_text_logging=False)

    turns, _used_references = adapter.calls[0]
    assert turns == [(0, "<English> Hello")]


@pytest.mark.unit
def test_single_character_native_call_uses_narrator_as_speaker_one_when_no_global_narrator_exists(monkeypatch):
    parser = FakeCharacterParser()

    def parse_single_bob(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(character="bob", text="Bonjour", parameters={}, language="fr", explicit_language=False),
        ]

    parser.parse_text_segments = parse_single_bob
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_mapping",
        lambda *_args, **_kwargs: {"bob": ("bob.wav", "Bob ref")},
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"id": "speaker1_female"}],
            "multi_speaker_mode": "Native Multi-Speaker",
            "language_prompting": "Off",
        },
    )

    processor.process_text(
        "subtitle",
        {"narrator": {"id": "narrator_voice"}},
        seed=1,
        reference_order=["bob"],
        show_text_logging=False,
    )

    turns, used_references = adapter.calls[0]
    assert turns == [(0, "Bonjour")]
    assert used_references == [{"id": "narrator_voice"}]


@pytest.mark.unit
def test_first_tagged_character_uses_speaker2_when_global_narrator_exists(monkeypatch):
    parser = FakeCharacterParser()

    def parse_single_bob(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(character="bob", text="Bonjour", parameters={}, language="fr", explicit_language=False),
        ]

    parser.parse_text_segments = parse_single_bob
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_mapping",
        lambda *_args, **_kwargs: {"bob": ("bob.wav", "Bob ref")},
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"id": "speaker2_voice"}],
            "multi_speaker_mode": "Native Multi-Speaker",
            "language_prompting": "Off",
        },
    )

    processor.process_text(
        "subtitle",
        {"narrator": {"id": "narrator_voice"}},
        seed=1,
        reference_order=["narrator", "bob"],
        show_text_logging=False,
    )

    turns, used_references = adapter.calls[0]
    assert turns == [(0, "Bonjour")]
    assert used_references == [{"id": "speaker2_voice"}]


@pytest.mark.unit
def test_second_character_uses_speaker2_override_when_narrator_not_reserved(monkeypatch):
    parser = FakeCharacterParser()

    def parse_bob_then_rick(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(character="bob", text="Bonjour", parameters={}, language="fr", explicit_language=False),
            SimpleNamespace(character="rick", text="Salut", parameters={}, language="fr", explicit_language=False),
        ]

    parser.parse_text_segments = parse_bob_then_rick
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_mapping",
        lambda *_args, **_kwargs: {"bob": ("bob.wav", "Bob ref"), "rick": ("rick.wav", "Rick ref")},
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"id": "speaker2_voice"}],
            "multi_speaker_mode": "Native Multi-Speaker",
            "language_prompting": "Off",
        },
    )

    processor.process_text(
        "subtitle",
        {"narrator": {"id": "narrator_voice"}},
        seed=1,
        reference_order=["bob", "rick"],
        show_text_logging=False,
    )

    turns, used_references = adapter.calls[0]
    assert turns == [(0, "Bonjour"), (1, "Salut")]
    assert used_references == [{"id": "narrator_voice"}, {"id": "speaker2_voice"}]


@pytest.mark.unit
def test_native_override_uses_effective_speaker_default_language_for_auto_inline_tag(monkeypatch):
    parser = FakeCharacterParser()

    def parse_single_bob(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(
                character="male_01",
                original_character="Bob",
                text="Bonjour",
                parameters={},
                language="fr",
                explicit_language=False,
            ),
        ]

    parser.parse_text_segments = parse_single_bob
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_mapping",
        lambda *_args, **_kwargs: {"male_01": ("bob.wav", "Bob ref")},
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_default_language",
        lambda character: {"anna": "de", "male_01": "fr"}.get(character.lower()),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"character_name": "Anna"}],
            "multi_speaker_mode": "Native Multi-Speaker",
            "language_prompting": "Auto Inline Tag",
        },
    )

    processor.process_text(
        "subtitle",
        {"narrator": {"character_name": "Joe"}},
        seed=1,
        reference_order=["narrator", "male_01"],
        show_text_logging=False,
    )

    turns, used_references = adapter.calls[0]
    assert turns == [(0, "<German> Bonjour")]
    assert used_references == [{"character_name": "Anna"}]


@pytest.mark.unit
def test_native_override_with_unmapped_connected_voice_does_not_fall_back_to_alias_language(monkeypatch):
    parser = FakeCharacterParser()

    def parse_single_bob(_text, engine_type=None):
        assert engine_type == "fish_audio_s2"
        return [
            SimpleNamespace(
                character="male_01",
                original_character="Bob",
                text="Bonjour",
                parameters={},
                language="fr",
                explicit_language=False,
            ),
        ]

    parser.parse_text_segments = parse_single_bob
    monkeypatch.setattr(PROCESSOR_MODULE, "character_parser", parser)
    monkeypatch.setattr(PROCESSOR_MODULE, "get_available_characters", lambda: [])
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_mapping",
        lambda *_args, **_kwargs: {"male_01": ("bob.wav", "Bob ref")},
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "get_character_default_language",
        lambda character: None,
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "model_management",
        SimpleNamespace(interrupt_processing=False),
    )
    monkeypatch.setattr(
        PROCESSOR_MODULE,
        "voice_discovery",
        SimpleNamespace(
            get_character_aliases=lambda: {},
            get_character_language_defaults=lambda: {},
        ),
    )

    adapter = CapturingAdapter()
    processor = PROCESSOR_MODULE.FishAudioS2Processor(
        adapter,
        {
            "speaker_references": [{"character_name": "Tony"}],
            "multi_speaker_mode": "Native Multi-Speaker",
            "language_prompting": "Auto Inline Tag",
        },
    )

    processor.process_text(
        "subtitle",
        {"narrator": {"character_name": "Joe"}},
        seed=1,
        reference_order=["narrator", "male_01"],
        show_text_logging=False,
    )

    turns, used_references = adapter.calls[0]
    assert turns == [(0, "Bonjour")]
    assert used_references == [{"character_name": "Tony"}]
