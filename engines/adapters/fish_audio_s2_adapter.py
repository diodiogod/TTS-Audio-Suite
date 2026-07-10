"""Adapter between suite processors and the isolated official Fish S2 runtime."""

from typing import Any, Dict, Optional

import torch

from engines.fish_audio_s2.downloader import FishAudioS2Downloader
from utils.audio.audio_hash import generate_stable_audio_component
from utils.audio.cache import get_audio_cache
from utils.audio.processing import AudioProcessingUtils
from utils.models.factory_config import ModelLoadConfig
from utils.text.fish_audio_s2_tags import translate_fish_s2_inline_tags
from utils.voice.character_logging import resolved_voice_name


class FishAudioS2Adapter:
    SAMPLE_RATE = 44100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self.audio_cache = get_audio_cache()
        self._model_config = None

    def update_config(self, config):
        self.config = dict(config or {})

    def _engine(self):
        from utils.models.unified_model_interface import unified_model_interface
        model_selection = self.config.get("model_variant", "s2-pro")
        quantization = self.config.get("quantization", "none")
        model_path = FishAudioS2Downloader.resolve_model_path(model_selection)
        model_variant = FishAudioS2Downloader.resolve_model_variant(model_selection, model_path)
        self._model_config = ModelLoadConfig(
            engine_name="fish_audio_s2", model_type="tts", model_name=model_variant,
            model_path=model_path, device=self.config.get("device", "auto"),
            runtime_mode="isolated",
            additional_params={
                "model_variant": model_variant,
                "quantization": quantization,
                "precision": self.config.get("precision", "bfloat16"),
                "compile": bool(self.config.get("compile", False)),
                "context_length": int(self.config.get("context_length", 8192)),
            },
        )
        return unified_model_interface.load_model(self._model_config)

    def _reference(self, voice_ref):
        if not isinstance(voice_ref, dict):
            return None, "", "default_voice"
        text = (voice_ref.get("reference_text") or voice_ref.get("prompt_text") or "").strip()
        path = voice_ref.get("audio_path") or voice_ref.get("prompt_audio_path")
        audio = voice_ref.get("audio")
        if audio is None:
            audio = voice_ref.get("waveform")
        if not path and isinstance(audio, dict):
            path = AudioProcessingUtils.save_audio_to_temp_file(audio["waveform"], audio.get("sample_rate", 44100))
        elif not path and torch.is_tensor(audio):
            path = AudioProcessingUtils.save_audio_to_temp_file(audio, voice_ref.get("sample_rate", 44100))
        component = generate_stable_audio_component(audio_file_path=path) if path else "default_voice"
        return path, text, component

    def generate_single(self, text, voice_ref, seed=0, enable_audio_cache=True, character_name=None):
        return self.generate_dialogue(
            [(0, text)], [voice_ref], seed, enable_audio_cache,
            cache_character=character_name or "narrator",
        )

    def generate_dialogue(self, turns, voice_refs, seed=0, enable_audio_cache=True,
                          cache_character="native_dialogue"):
        formatted_turns = []
        for speaker_index, turn_text in turns:
            clean_text = translate_fish_s2_inline_tags((turn_text or "").strip())
            if clean_text:
                formatted_turns.append(f"<|speaker:{speaker_index}|>{clean_text}")
        text = "\n".join(formatted_turns)
        if not text:
            return torch.zeros(1, 0)

        references = []
        reference_labels = []
        components = []
        reference_texts = []
        for speaker_index, voice_ref in enumerate(voice_refs):
            ref_path, ref_text, component = self._reference(voice_ref)
            if ref_path:
                if not ref_text:
                    raise ValueError("Fish S2 native speakers require exact reference transcripts")
                references.append({
                    "audio_path": ref_path,
                    "text": f"<|speaker:{speaker_index}|>{ref_text}",
                })
                reference_labels.append(
                    f"local Speaker {speaker_index + 1}={resolved_voice_name(voice_ref)}"
                )
            components.append(component)
            reference_texts.append(ref_text)
        if references:
            print(f"🎤 Fish local reference order: {', '.join(reference_labels)}")

        params = {
            "model_variant": self.config.get("model_variant", "s2-pro"),
            "quantization": self.config.get("quantization", "none"),
            "multi_speaker_mode": self.config.get("multi_speaker_mode", "Native Multi-Speaker"),
            "seed": int(seed), "normalize": bool(self.config.get("normalize", True)),
            "chunk_length": int(self.config.get("native_chunk_length", 200)),
            "max_new_tokens": int(self.config.get("max_new_tokens", 1024)),
            "top_p": float(self.config.get("top_p", 0.8)),
            "repetition_penalty": float(self.config.get("repetition_penalty", 1.1)),
            "temperature": float(self.config.get("temperature", 0.8)),
            "cache_reference": bool(self.config.get("cache_reference", True)),
            "context_length": int(self.config.get("context_length", 8192)),
        }
        cache_key = self.audio_cache.generate_cache_key(
            "fish_audio_s2", text=text, audio_component="|".join(components),
            reference_text="|".join(reference_texts), character=cache_character, **params,
        ) if enable_audio_cache else None
        if cache_key:
            cached = self.audio_cache.get_cached_audio(cache_key)
            if cached:
                print("💾 Fish Audio S2: Using cached audio")
                return cached[0]
        audio, sample_rate = self._engine().generate(text=text, references=references, **params)
        if sample_rate != self.SAMPLE_RATE:
            raise RuntimeError(f"Fish S2 returned unexpected sample rate {sample_rate}")
        audio = audio.detach().float().cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if cache_key:
            self.audio_cache.cache_audio(cache_key, audio, audio.shape[-1] / self.SAMPLE_RATE)
        return audio
