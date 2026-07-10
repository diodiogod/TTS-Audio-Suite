import pytest

from utils.text.fish_audio_s2_tags import translate_fish_s2_inline_tags


@pytest.mark.unit
@pytest.mark.parametrize("source, expected", [
    ("Hello <whisper>world", "Hello [whisper]world"),
    ("<professional broadcast tone>Hello", "[professional broadcast tone]Hello"),
    ("<pitch up>now <angry>stop", "[pitch up]now [angry]stop"),
    ("[Alice] Hello <excited>", "[Alice] Hello [excited]"),
    ("Keep <unclosed", "Keep <unclosed"),
])
def test_translates_free_form_fish_tags_at_boundary(source, expected):
    assert translate_fish_s2_inline_tags(source) == expected


@pytest.mark.unit
def test_normalizes_only_internal_whitespace():
    assert translate_fish_s2_inline_tags("<  whisper   in small voice  >") == "[whisper in small voice]"
