"""audio.cpp adapter for the Suite's unified Voice Changer node."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping

import torch

from engines.adapters.audio_cpp_adapter import AudioCppEngineAdapter
from utils.audio.processing import AudioProcessingUtils


def _advanced_options(config: Mapping[str, Any]) -> Dict[str, Any]:
    value = config.get("advanced_options", config.get("request_options", {}))
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid audio.cpp advanced JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("audio.cpp advanced options must be a JSON object")
    return dict(value)


def _materialize(audio: Mapping[str, Any], label: str) -> str:
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not torch.is_tensor(waveform):
        raise TypeError(f"audio.cpp {label} must contain a waveform tensor")
    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError(f"audio.cpp {label} must contain a positive sample_rate")
    return os.path.abspath(
        AudioProcessingUtils.save_audio_to_temp_file(waveform, int(sample_rate))
    )


class AudioCppVoiceConversionAdapter:
    """Convert source audio toward a target reference using an audio.cpp VC task."""

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)

    def _session_config(self) -> Dict[str, Any]:
        config = dict(self.config)
        if str(config.get("connection_mode", "auto")).lower() != "external_server":
            config["requested_task"] = "vc"
            config["task"] = "vc"
        return config

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        target_audio: Dict[str, Any],
        refinement_passes: int = 1,
    ) -> tuple[Dict[str, Any], str]:
        from utils.audio_cpp.session import get_audio_cpp_session

        config = self._session_config()
        family = str(config.get("family", "")).strip()
        passes = max(1, int(refinement_passes))
        current = source_audio
        output_rate = int(source_audio["sample_rate"])

        session = get_audio_cpp_session(config)
        if str(getattr(session, "task", "vc")) != "vc":
            raise ValueError(
                f"audio.cpp model '{session.model_id}' is configured for task "
                f"'{session.task}', not voice conversion"
            )

        for pass_index in range(passes):
            source_path = _materialize(current, "source audio")
            target_path = _materialize(target_audio, "target reference audio")
            try:
                request = {
                    "audio": source_path,
                    "voice_ref": target_path,
                    "source_audio": source_path,
                    "target_voice": target_path,
                    "options": _advanced_options(config),
                }
                print(
                    f"🔄 audio.cpp VC: {family or 'external model'} pass "
                    f"{pass_index + 1}/{passes}..."
                )
                result = session.run(request)
                waveform, output_rate = AudioCppEngineAdapter._normalize_result(result)
                current = {"waveform": waveform.unsqueeze(0), "sample_rate": output_rate}
            finally:
                for path in (source_path, target_path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

        info = (
            f"Model family: {family or getattr(session, 'family', 'external')}\n"
            f"Model ID: {session.model_id}\n"
            f"Task: voice conversion\n"
            f"Refinement passes: {passes}\n"
            f"Output sample rate: {output_rate} Hz\n"
            "Conversion completed successfully"
        )
        return current, info


__all__ = ["AudioCppVoiceConversionAdapter"]
