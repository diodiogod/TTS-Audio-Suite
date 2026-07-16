"""Provider registry for reference-free voice design workflows."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch


@dataclass
class VoiceDesignResult:
    opt_narrator: Dict[str, Any]
    preview_audio: Dict[str, Any]
    voice_info: str


class VoiceDesignerProvider(Protocol):
    def design(
        self,
        engine_data: Dict[str, Any],
        voice_instruction: str,
        reference_text: str,
        seed: int,
    ) -> VoiceDesignResult: ...


def _engine_config(engine_data: Dict[str, Any]) -> Dict[str, Any]:
    config = engine_data.get("config", engine_data)
    if not isinstance(config, dict):
        raise TypeError("TTS_ENGINE config must be a dictionary")
    return config


def _engine_language(config: Dict[str, Any]) -> str:
    return str(config.get("language") or "Auto").strip() or "Auto"


def _generation_fingerprint(
    engine_type: str,
    config: Dict[str, Any],
    reference_text: str,
    seed: int,
) -> str | None:
    """Identify a reproducible voice-design generation.

    Seed 0 deliberately keeps each provider's random behavior, so two runs with
    otherwise identical inputs are not assumed to produce the same character.
    """
    if int(seed or 0) == 0:
        return None

    payload = {
        "schema": 1,
        "engine": engine_type,
        "config": config,
        "reference_text": reference_text,
        "seed": int(seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _attach_generation_fingerprint(
    opt_narrator: Dict[str, Any],
    engine_type: str,
    config: Dict[str, Any],
    reference_text: str,
    seed: int,
) -> None:
    fingerprint = _generation_fingerprint(engine_type, config, reference_text, seed)
    if fingerprint:
        opt_narrator["generation_fingerprint"] = fingerprint


def _audio_output(waveform: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
    waveform = waveform.detach().float().cpu()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 3:
        raise ValueError(f"Voice designer returned unsupported waveform shape: {tuple(waveform.shape)}")
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _narrator(
    audio: Dict[str, Any],
    reference_text: str,
    instruction: str,
    engine: str,
    model: str,
    language: str,
) -> Dict[str, Any]:
    return {
        "audio": audio,
        "audio_path": None,
        "reference_text": reference_text,
        "character_name": "voice_design",
        "source": "voice_design",
        "description": instruction,
        "design_instruction": instruction,
        "language": language,
        "engine": engine,
        "model": model,
    }


class QwenVoiceDesignerProvider:
    _node_class = None

    @classmethod
    def _load_node_class(cls):
        if cls._node_class is not None:
            return cls._node_class
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        module_path = os.path.join(project_root, "nodes", "qwen3_tts", "qwen3_tts_voice_designer_node.py")
        module_name = "tts_audio_suite_qwen_voice_designer_provider"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Qwen voice designer from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cls._node_class = module.Qwen3TTSVoiceDesignerNode
        return cls._node_class

    def design(self, engine_data, voice_instruction, reference_text, seed):
        config = dict(_engine_config(engine_data))
        if config.get("model_role") != "voice_design" or config.get("model_type") != "VoiceDesign":
            selected = config.get("model_variant") or config.get("model_size") or "unknown"
            raise ValueError(
                f"Qwen model '{selected}' cannot design voices. In the Qwen3-TTS Engine, "
                "select 'Voice Design - 1.7B VoiceDesign', then run the workflow again."
            )
        instruction = str(voice_instruction or "").strip()
        language = _engine_language(config)
        config["instruct"] = instruction
        node = self._load_node_class()()
        opt_narrator, audio, info = node.design_voice(
            engine_data, instruction, reference_text, language,
            character_name="", overwrite_character=False, seed=seed,
        )
        opt_narrator.update({
            "design_instruction": instruction,
            "model": config.get("model_name", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        })
        _attach_generation_fingerprint(opt_narrator, "qwen3_tts", config, reference_text, seed)
        info = info.replace("\n\n💡 Tip: Provide a character_name to save this voice for reuse", "")
        return VoiceDesignResult(opt_narrator, audio, info)


class OmniVoiceDesignerProvider:
    def design(self, engine_data, voice_instruction, reference_text, seed):
        from engines.adapters.omnivoice_adapter import OmniVoiceEngineAdapter

        config = dict(_engine_config(engine_data))
        if config.get("model_role") != "voice_design":
            raise ValueError(
                "OmniVoice is configured for Text to Speech. Set mode to 'Voice Design' in the "
                "OmniVoice Engine, then run the workflow again."
            )
        instruction = str(voice_instruction or "").strip()
        config["instruct"] = instruction
        language = _engine_language(config)
        adapter = OmniVoiceEngineAdapter(config)
        waveform = adapter.generate_single(
            reference_text, voice_ref=None, seed=seed,
            enable_audio_cache=True, character_name="voice_design",
        )
        audio = _audio_output(waveform, adapter.SAMPLE_RATE)
        model = str(config.get("model_variant", "OmniVoice"))
        opt_narrator = _narrator(audio, reference_text, instruction, "omnivoice", model, language)
        _attach_generation_fingerprint(opt_narrator, "omnivoice", config, reference_text, seed)
        info = (
            "Voice designed through reference-free OmniVoice TTS generation\n"
            f"Instruction: {instruction}\nModel: {model}"
        )
        return VoiceDesignResult(opt_narrator, audio, info)


class MossVoiceDesignerProvider:
    MODEL_NAME = "MOSS-VoiceGenerator"

    def design(self, engine_data, voice_instruction, reference_text, seed):
        from engines.adapters.moss_tts_adapter import MossTTSEngineAdapter
        from engines.moss_tts.model_specs import MOSS_MODEL_SPECS

        config = dict(_engine_config(engine_data))
        selected_model = str(config.get("model_variant") or "")
        if config.get("model_role") != "voice_design" or "MOSS-VoiceGenerator" not in selected_model:
            raise ValueError(
                f"MOSS model '{selected_model or 'unknown'}' cannot design voices. In the MOSS-TTS Engine, "
                "select 'Voice Design 1.7B (MOSS-VoiceGenerator)', then run the workflow again."
            )
        instruction = str(voice_instruction or "").strip()
        config["instruction"] = instruction
        spec = MOSS_MODEL_SPECS[self.MODEL_NAME]
        resolved_language = _engine_language(config)
        adapter = MossTTSEngineAdapter()
        adapter.load_model(
            model_variant=selected_model,
            device=config.get("device", "auto"),
            dtype=config.get("dtype", "auto"),
            attn_implementation=config.get("attn_implementation", "auto"),
            codec_model=spec["codec_model"],
        )
        params = {
            "model_variant": selected_model,
            "language": resolved_language,
            "instruction": instruction,
            "seed": int(seed or 0),
            "device": config.get("device", "auto"),
            "dtype": config.get("dtype", "auto"),
            "attn_implementation": config.get("attn_implementation", "auto"),
            "audio_temperature": spec["audio_temperature"],
            "audio_top_p": spec["audio_top_p"],
            "audio_top_k": spec["audio_top_k"],
            "audio_repetition_penalty": spec["audio_repetition_penalty"],
            "max_new_tokens": spec["max_new_tokens"],
        }
        waveform = adapter.generate_with_pause_tags(
            reference_text, voice_ref=None, params=params,
            process_pauses=False, character_name="voice_design",
        )
        audio = _audio_output(waveform, int(spec["sample_rate"]))
        opt_narrator = _narrator(
            audio, reference_text, instruction, "moss_tts",
            selected_model, resolved_language,
        )
        _attach_generation_fingerprint(opt_narrator, "moss_tts", config, reference_text, seed)
        info = f"Voice designed with MOSS VoiceGenerator\nInstruction: {instruction}\nModel: {selected_model}"
        return VoiceDesignResult(opt_narrator, audio, info)


_PROVIDERS: Dict[str, VoiceDesignerProvider] = {
    "qwen3_tts": QwenVoiceDesignerProvider(),
    "moss_tts": MossVoiceDesignerProvider(),
    "omnivoice": OmniVoiceDesignerProvider(),
}


def design_voice(
    engine_data: Dict[str, Any],
    voice_instruction: str,
    reference_text: str,
    seed: int = 0,
) -> VoiceDesignResult:
    if not isinstance(engine_data, dict):
        raise TypeError("Voice Designer requires a TTS_ENGINE connection")
    capabilities = engine_data.get("capabilities") or []
    if "voice_design" not in capabilities:
        raise ValueError(
            "The connected engine is not configured for voice design. Select a VoiceDesign model or "
            "Voice Design mode in the engine node, then run the workflow again."
        )
    engine_type = str(engine_data.get("engine_type") or engine_data.get("config", {}).get("engine_type") or "")
    provider = _PROVIDERS.get(engine_type)
    if provider is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Engine '{engine_type or 'unknown'}' cannot design voices. Supported engines: {supported}")
    if len(str(reference_text or "").strip()) < 10:
        raise ValueError("Reference text must contain at least 10 characters")
    clean_instruction = str(voice_instruction or "").strip()
    if not clean_instruction:
        raise ValueError("Voice design instruction cannot be empty")
    return provider.design(
        engine_data,
        clean_instruction,
        str(reference_text).strip(),
        int(seed or 0),
    )
