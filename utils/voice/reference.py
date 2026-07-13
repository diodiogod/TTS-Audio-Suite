"""Shared voice-reference resolution rules.

Processed tensor audio is authoritative. File paths are fallbacks and may point to
the unprocessed source used to derive a trimmed or otherwise customized reference.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple


DEFAULT_AUDIO_KEYS = ("audio", "waveform", "audio_dict", "reference_audio")
DEFAULT_PATH_KEYS = ("prompt_audio_path", "audio_path", "waveform_path")


def resolve_effective_voice_audio(
    voice_ref: Optional[Mapping[str, Any]],
    *,
    audio_keys: Sequence[str] = DEFAULT_AUDIO_KEYS,
    path_keys: Sequence[str] = DEFAULT_PATH_KEYS,
) -> Tuple[Any, Optional[str]]:
    """Return ``(effective_audio, source_key)`` using tensor-first precedence.

    Non-string values under audio keys are treated as processed audio. String
    values under those keys remain supported, but only after explicit path keys.
    """
    if not isinstance(voice_ref, Mapping):
        return None, None

    string_audio_fallbacks = []
    for key in audio_keys:
        value = voice_ref.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            string_audio_fallbacks.append((value, key))
        else:
            return value, key

    for key in path_keys:
        value = voice_ref.get(key)
        if value is not None and value != "":
            return value, key

    if string_audio_fallbacks:
        return string_audio_fallbacks[0]
    return None, None


def effective_voice_audio(
    voice_ref: Optional[Mapping[str, Any]],
    *,
    audio_keys: Sequence[str] = DEFAULT_AUDIO_KEYS,
    path_keys: Sequence[str] = DEFAULT_PATH_KEYS,
) -> Any:
    """Return only the effective audio value."""
    return resolve_effective_voice_audio(
        voice_ref,
        audio_keys=audio_keys,
        path_keys=path_keys,
    )[0]
