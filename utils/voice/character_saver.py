"""Shared character-voice persistence for TTS Audio Suite.

This module owns the established character format:

    name.wav
    name.reference.txt
    name.txt

Nodes should pass an existing ``NARRATOR_VOICE``/``opt_narrator`` dictionary
here instead of implementing their own filesystem writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional

import torch

from utils.voice.reference import effective_voice_audio


_INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class CharacterSaveResult:
    opt_narrator: Dict[str, Any]
    character_name: str
    audio_path: str
    reference_path: str
    metadata_path: str
    info: str


def _validate_path_part(value: str, label: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{label} cannot be empty")
    if value in {".", ".."} or value.endswith((" ", ".")):
        raise ValueError(f"Invalid {label}: '{value}'")
    if _INVALID_WINDOWS_CHARS.search(value):
        raise ValueError(f"Invalid {label}: '{value}' contains a reserved character")
    if value.lower() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Invalid {label}: '{value}' is a reserved Windows name")
    return value


def _resolve_audio(opt_narrator: Dict[str, Any]) -> Dict[str, Any]:
    audio = effective_voice_audio(opt_narrator)
    if isinstance(audio, dict) and audio.get("waveform") is not None:
        return {
            "waveform": audio["waveform"],
            "sample_rate": int(audio.get("sample_rate", 0) or 0),
        }

    if torch.is_tensor(audio):
        sample_rate = int(opt_narrator.get("sample_rate", 0) or 0)
        return {"waveform": audio, "sample_rate": sample_rate}

    if isinstance(audio, (str, os.PathLike)):
        from utils.audio.processing import AudioProcessingUtils

        waveform, sample_rate = AudioProcessingUtils.safe_load_audio(str(audio))
        return {"waveform": waveform, "sample_rate": int(sample_rate)}

    raise ValueError("opt_narrator does not contain usable audio")


def _normalize_waveform(audio: Dict[str, Any]) -> tuple[torch.Tensor, int]:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if not torch.is_tensor(waveform) or waveform.numel() == 0:
        raise ValueError("Character voice audio is empty")
    if sample_rate <= 0:
        raise ValueError("Character voice audio has no valid sample rate")

    if waveform.dim() == 3:
        if waveform.shape[0] != 1:
            raise ValueError("Save Character Voice accepts one audio item, not an audio batch")
        waveform = waveform.squeeze(0)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 2:
        raise ValueError(f"Unsupported character voice waveform shape: {tuple(waveform.shape)}")
    return waveform.detach().float().cpu().contiguous(), sample_rate


def _metadata_text(opt_narrator: Dict[str, Any], reference_text: str) -> str:
    lines = ["Voice saved by TTS Audio Suite", ""]
    values = (
        ("Source", opt_narrator.get("source")),
        ("Engine", opt_narrator.get("engine")),
        ("Model", opt_narrator.get("model") or opt_narrator.get("model_name")),
        ("Language", opt_narrator.get("language")),
    )
    for label, value in values:
        if value is not None and str(value).strip():
            lines.append(f"{label}: {str(value).strip()}")

    instruction = (
        opt_narrator.get("design_instruction")
        or opt_narrator.get("description")
        or opt_narrator.get("voice_description")
    )
    if instruction and str(instruction).strip():
        lines.extend(["", "Voice Description:", str(instruction).strip()])

    lines.extend(["", "Reference Text:", reference_text])
    return "\n".join(lines).rstrip() + "\n"


def _choose_character_name(target_dir: Path, requested_name: str, overwrite_character: bool) -> str:
    def exists(name: str) -> bool:
        return any((target_dir / f"{name}{suffix}").exists() for suffix in (".wav", ".reference.txt", ".txt"))

    if not exists(requested_name) or overwrite_character:
        return requested_name

    counter = 1
    while exists(f"{requested_name}_{counter}"):
        counter += 1
    return f"{requested_name}_{counter}"


def _notify_voice_library_changed(character_name: str) -> None:
    from utils.voice.discovery import get_available_characters, get_available_voices

    get_available_voices(force_refresh=True)
    get_available_characters(force_refresh=True)
    try:
        from server import PromptServer

        if PromptServer.instance is not None:
            PromptServer.instance.send_sync(
                "tts-audio-suite.voice-library-changed",
                {"character_name": character_name},
            )
    except Exception:
        # Saving must remain usable in tests and non-server contexts.
        pass


def save_character_voice(
    opt_narrator: Dict[str, Any],
    character_name: str,
    overwrite_character: bool = False,
    metadata_text: Optional[str] = None,
) -> CharacterSaveResult:
    """Persist one opt_narrator in the shared character-voice library."""
    if not isinstance(opt_narrator, dict):
        raise TypeError("Save Character Voice requires an opt_narrator/NARRATOR_VOICE input")

    safe_name = _validate_path_part(character_name, "character name")
    reference_text = str(opt_narrator.get("reference_text") or "").strip()
    if not reference_text:
        raise ValueError(
            "Save Character Voice requires the exact spoken transcription in opt_narrator.reference_text"
        )

    audio = _resolve_audio(opt_narrator)
    waveform, sample_rate = _normalize_waveform(audio)

    import folder_paths

    voices_root = Path(folder_paths.models_dir).resolve() / "voices"
    target_dir = voices_root
    target_dir.mkdir(parents=True, exist_ok=True)

    final_name = _choose_character_name(target_dir, safe_name, overwrite_character)
    audio_path = target_dir / f"{final_name}.wav"
    reference_path = target_dir / f"{final_name}.reference.txt"
    metadata_path = target_dir / f"{final_name}.txt"
    resolved_metadata = metadata_text if metadata_text is not None else _metadata_text(opt_narrator, reference_text)

    temp_paths: list[Path] = []
    try:
        import torchaudio

        audio_fd, audio_tmp = tempfile.mkstemp(prefix=f".{final_name}.", suffix=".wav", dir=target_dir)
        os.close(audio_fd)
        audio_temp_path = Path(audio_tmp)
        temp_paths.append(audio_temp_path)
        torchaudio.save(str(audio_temp_path), waveform, sample_rate, format="wav")

        ref_fd, ref_tmp = tempfile.mkstemp(prefix=f".{final_name}.", suffix=".reference.txt", dir=target_dir)
        with os.fdopen(ref_fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(reference_text)
        reference_temp_path = Path(ref_tmp)
        temp_paths.append(reference_temp_path)

        meta_fd, meta_tmp = tempfile.mkstemp(prefix=f".{final_name}.", suffix=".txt", dir=target_dir)
        with os.fdopen(meta_fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(str(resolved_metadata).rstrip() + "\n")
        metadata_temp_path = Path(meta_tmp)
        temp_paths.append(metadata_temp_path)

        os.replace(audio_temp_path, audio_path)
        os.replace(reference_temp_path, reference_path)
        os.replace(metadata_temp_path, metadata_path)
    finally:
        for temp_path in temp_paths:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    updated = dict(opt_narrator)
    updated.update(
        {
            "audio": {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},
            "audio_path": str(audio_path),
            "source_audio_path": str(audio_path),
            "reference_text": reference_text,
            "canonical_reference_text": reference_text,
            "character_name": final_name,
            "source": "character_library",
            "customized": False,
        }
    )

    _notify_voice_library_changed(final_name)
    info = f"Saved character '{final_name}' to models/voices"
    return CharacterSaveResult(
        opt_narrator=updated,
        character_name=final_name,
        audio_path=str(audio_path),
        reference_path=str(reference_path),
        metadata_path=str(metadata_path),
        info=info,
    )
