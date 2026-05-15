"""
MOSS-TTS engine adapter.

Translates TTS Audio Suite processor calls into official MOSS-TTS inference
without exposing non-native controls.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.models.factory_config import ModelLoadConfig
from utils.text.pause_processor import PauseTagProcessor


class MossTTSEngineAdapter:
    """Adapter for the official MOSS-TTS model API."""

    SAMPLE_RATE = 24000
    MOSS_TOKENS_PER_SECOND = 12.5

    def __init__(self, node_instance=None):
        self.node = node_instance
        self.audio_cache = get_audio_cache()
        self._last_config: Optional[ModelLoadConfig] = None

    def load_model(
        self,
        model_variant: str,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "auto",
        codec_model: str = "MOSS-Audio-Tokenizer",
    ):
        from utils.models.unified_model_interface import unified_model_interface

        config = ModelLoadConfig(
            engine_name="moss_tts",
            model_type="tts",
            model_name=model_variant,
            model_path=model_variant,
            device=device,
            additional_params={
                "dtype": dtype,
                "attn_implementation": attn_implementation,
                "codec_model": codec_model,
            },
        )
        self._last_config = config
        unified_model_interface.load_model(config)

    def update_model_config(
        self,
        model_variant: str,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "auto",
        codec_model: str = "MOSS-Audio-Tokenizer",
    ):
        self.load_model(model_variant, device, dtype, attn_implementation, codec_model)

    def _get_engine(self):
        if self._last_config is None:
            raise RuntimeError("MOSS-TTS model has not been loaded")
        from utils.models.unified_model_interface import unified_model_interface

        return unified_model_interface.load_model(self._last_config)

    def generate_with_pause_tags(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        process_pauses: bool = True,
        character_name: Optional[str] = None,
    ) -> torch.Tensor:
        engine = self._get_engine()
        seed = int(params.get("seed", 0) or 0)
        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if process_pauses and PauseTagProcessor.has_pause_tags(text):
            return self._generate_with_pauses(text, voice_ref, params, character_name, engine)
        return self._generate_direct(text, voice_ref, params, character_name, engine)

    def _generate_direct(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        character_name: Optional[str] = None,
        engine=None,
    ) -> torch.Tensor:
        if engine is None:
            engine = self._get_engine()

        reference_audio, reference_sample_rate, audio_component = self._extract_voice_reference(voice_ref)
        model_variant = params.get("model_variant", "MOSS-TTS-Local-Transformer")
        language = params.get("language", "auto")
        duration_tokens, resolved_max_new_tokens = self._resolve_generation_lengths(text, params)
        n_vq_for_inference = params.get("n_vq_for_inference")

        cache_key = self.audio_cache.generate_cache_key(
            "moss_tts",
            text=text,
            model_variant=model_variant,
            audio_component=audio_component,
            language=language,
            duration_tokens=duration_tokens,
            audio_temperature=params.get("audio_temperature", params.get("temperature", 1.0)),
            audio_top_p=params.get("audio_top_p", params.get("top_p", 0.95)),
            audio_top_k=params.get("audio_top_k", params.get("top_k", 50)),
            audio_repetition_penalty=params.get("audio_repetition_penalty", params.get("repetition_penalty", 1.1)),
            max_new_tokens=resolved_max_new_tokens,
            n_vq_for_inference=n_vq_for_inference,
            seed=params.get("seed", 0),
            device=params.get("device", "auto"),
            dtype=params.get("dtype", "auto"),
            attn_implementation=params.get("attn_implementation", "auto"),
            character=character_name or "narrator",
        )

        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            print(f"💾 Using cached MOSS-TTS audio for '{character_name or 'narrator'}': '{text[:30]}...'")
            return cached_audio[0]

        audio_tensor, sample_rate = engine.generate(
            text=text,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            language=language,
            duration_tokens=duration_tokens,
            seed=int(params.get("seed", 0) or 0),
            audio_temperature=float(params.get("audio_temperature", params.get("temperature", 1.0))),
            audio_top_p=float(params.get("audio_top_p", params.get("top_p", 0.95))),
            audio_top_k=int(params.get("audio_top_k", params.get("top_k", 50))),
            audio_repetition_penalty=float(
                params.get("audio_repetition_penalty", params.get("repetition_penalty", 1.1))
            ),
            max_new_tokens=int(resolved_max_new_tokens),
            n_vq_for_inference=n_vq_for_inference,
        )

        if sample_rate != self.SAMPLE_RATE:
            raise RuntimeError(f"MOSS-TTS returned unexpected sample rate {sample_rate}; expected {self.SAMPLE_RATE}")

        duration = self.audio_cache._calculate_duration(audio_tensor, "moss_tts")
        self.audio_cache.cache_audio(cache_key, audio_tensor, duration)
        return audio_tensor

    def _resolve_generation_lengths(self, text: str, params: Dict[str, Any]) -> Tuple[Optional[int], int]:
        explicit_duration = params.get("duration_tokens")
        requested_max = int(params.get("max_new_tokens", 4096))

        if explicit_duration:
            duration_tokens = max(1, int(explicit_duration))
            resolved_max = max(requested_max, duration_tokens + 8)
            return duration_tokens, resolved_max

        estimated_duration = self._estimate_duration_tokens(text)
        resolved_max = min(requested_max, max(estimated_duration + 12, 24))

        if requested_max != resolved_max:
            print(
                f"⏱️ MOSS-TTS: Auto duration hint {estimated_duration} tokens, "
                f"resolved max_new_tokens={resolved_max} from requested {requested_max}"
            )
        else:
            print(f"⏱️ MOSS-TTS: Auto duration hint {estimated_duration} tokens")

        return estimated_duration, resolved_max

    def _estimate_duration_tokens(self, text: str) -> int:
        clean_text = re.sub(r"\s+", " ", str(text or "").strip())
        if not clean_text:
            return 24

        words = re.findall(r"\b[\w']+\b", clean_text)
        non_space_chars = len(re.sub(r"\s+", "", clean_text))

        seconds_from_words = len(words) / 2.6 if words else 0.0
        seconds_from_chars = non_space_chars / 13.5
        estimated_seconds = max(seconds_from_words, seconds_from_chars, 1.5)

        estimated_tokens = int(round(estimated_seconds * self.MOSS_TOKENS_PER_SECOND))
        return max(24, min(estimated_tokens, 2048))

    def _generate_with_pauses(
        self,
        text: str,
        voice_ref: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        character_name: Optional[str] = None,
        engine=None,
    ) -> torch.Tensor:
        segments, _ = PauseTagProcessor.parse_pause_tags(text)
        audio_parts = []
        for segment_type, content in segments:
            if segment_type == "text":
                audio_tensor = self._generate_direct(content, voice_ref, params, character_name, engine)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                elif audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_parts.append(audio_tensor)
            elif segment_type == "pause":
                audio_parts.append(PauseTagProcessor.create_silence_segment(content, self.SAMPLE_RATE))

        if not audio_parts:
            return torch.zeros(1, 0)
        return torch.cat(audio_parts, dim=-1)

    def _extract_voice_reference(self, voice_ref: Optional[Dict[str, Any]]) -> Tuple[Any, Optional[int], str]:
        if not voice_ref or not isinstance(voice_ref, dict):
            return None, None, "default_voice"

        ref_audio = (
            voice_ref.get("audio_path")
            or voice_ref.get("prompt_audio_path")
            or voice_ref.get("audio")
            or voice_ref.get("waveform")
        )
        if ref_audio is None:
            return None, None, "default_voice"

        if isinstance(ref_audio, str):
            return ref_audio, None, generate_stable_audio_component(audio_file_path=ref_audio)

        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            return (
                ref_audio,
                int(ref_audio.get("sample_rate", self.SAMPLE_RATE)),
                generate_stable_audio_component(reference_audio=ref_audio),
            )

        if torch.is_tensor(ref_audio):
            sample_rate = int(voice_ref.get("sample_rate", self.SAMPLE_RATE))
            audio_dict = {"waveform": ref_audio, "sample_rate": sample_rate}
            return audio_dict, sample_rate, generate_stable_audio_component(reference_audio=audio_dict)

        return ref_audio, voice_ref.get("sample_rate"), "custom_voice"

    def _normalize_dialogue_voice_ref(self, voice_ref: Any, speaker_label: str) -> Tuple[Optional[Dict[str, Any]], str]:
        if voice_ref is None:
            return None, "no_voice"

        if isinstance(voice_ref, str):
            print(
                f"❌ MOSS-TTSD {speaker_label}: received character name '{voice_ref}' instead of voice reference. "
                "Connect Character Voices opt_narrator, not character_name."
            )
            return None, "invalid_string_voice"

        if not isinstance(voice_ref, dict):
            print(f"❌ MOSS-TTSD {speaker_label}: invalid voice reference type {type(voice_ref).__name__}")
            return None, "invalid_voice"

        ref_audio = (
            voice_ref.get("audio_path")
            or voice_ref.get("prompt_audio_path")
            or voice_ref.get("audio")
            or voice_ref.get("waveform")
        )
        reference_text = (
            voice_ref.get("reference_text")
            or voice_ref.get("text")
            or voice_ref.get("prompt_text")
            or ""
        )

        if ref_audio is None:
            return None, "no_audio"

        normalized = {
            "audio": ref_audio,
            "sample_rate": voice_ref.get("sample_rate", self.SAMPLE_RATE),
            "reference_text": str(reference_text or "").strip(),
        }

        if isinstance(ref_audio, str):
            audio_component = generate_stable_audio_component(audio_file_path=ref_audio)
        elif isinstance(ref_audio, dict) and "waveform" in ref_audio:
            audio_component = generate_stable_audio_component(reference_audio=ref_audio)
            normalized["sample_rate"] = int(ref_audio.get("sample_rate", self.SAMPLE_RATE))
        elif torch.is_tensor(ref_audio):
            audio_component = generate_stable_audio_component(
                reference_audio={
                    "waveform": ref_audio,
                    "sample_rate": normalized["sample_rate"],
                }
            )
        else:
            audio_component = "custom_voice"

        if not normalized["reference_text"]:
            print(f"⚠️ MOSS-TTSD {speaker_label}: reference audio has no transcript; cloning disabled for this speaker")
            return None, audio_component

        return normalized, audio_component

    def generate_native_dialogue(
        self,
        dialogue_text: str,
        speaker_voices: List[Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        engine = self._get_engine()
        normalized_refs: List[Optional[Dict[str, Any]]] = []
        audio_components = []

        for idx, voice_ref in enumerate(speaker_voices):
            normalized, audio_component = self._normalize_dialogue_voice_ref(voice_ref, f"S{idx + 1}")
            normalized_refs.append(normalized)
            audio_components.append(audio_component)

        model_variant = params.get("model_variant", "MOSS-TTSD-v1.0")
        cache_key = self.audio_cache.generate_cache_key(
            "moss_tts",
            text=dialogue_text,
            model_variant=model_variant,
            audio_component="ttsd_" + "_".join(audio_components),
            language=params.get("language", "auto"),
            audio_temperature=params.get("audio_temperature", params.get("temperature", 1.1)),
            audio_top_p=params.get("audio_top_p", params.get("top_p", 0.9)),
            audio_top_k=params.get("audio_top_k", params.get("top_k", 50)),
            audio_repetition_penalty=params.get("audio_repetition_penalty", params.get("repetition_penalty", 1.1)),
            max_new_tokens=params.get("max_new_tokens", 4096),
            seed=params.get("seed", 0),
            device=params.get("device", "auto"),
            dtype=params.get("dtype", "auto"),
            attn_implementation=params.get("attn_implementation", "auto"),
            character="native_multispeaker",
        )

        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            print("💾 Using cached MOSS-TTSD native dialogue audio")
            audio_tensor = cached_audio[0]
        else:
            audio_tensor, sample_rate = engine.generate_dialogue(
                dialogue_text=dialogue_text,
                speaker_references=normalized_refs,
                language=params.get("language", "auto"),
                seed=int(params.get("seed", 0) or 0),
                audio_temperature=float(params.get("audio_temperature", params.get("temperature", 1.1))),
                audio_top_p=float(params.get("audio_top_p", params.get("top_p", 0.9))),
                audio_top_k=int(params.get("audio_top_k", params.get("top_k", 50))),
                audio_repetition_penalty=float(
                    params.get("audio_repetition_penalty", params.get("repetition_penalty", 1.1))
                ),
                max_new_tokens=int(params.get("max_new_tokens", 4096)),
            )

            if sample_rate != self.SAMPLE_RATE:
                raise RuntimeError(f"MOSS-TTSD returned unexpected sample rate {sample_rate}; expected {self.SAMPLE_RATE}")

            duration = self.audio_cache._calculate_duration(audio_tensor, "moss_tts")
            self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)

        return {
            "waveform": audio_tensor,
            "sample_rate": self.SAMPLE_RATE,
            "character": "native_multispeaker",
            "text": dialogue_text,
        }

    def cleanup(self):
        self._last_config = None
