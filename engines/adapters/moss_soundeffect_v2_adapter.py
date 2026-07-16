"""Suite adapter for MOSS-SoundEffect v2."""

from typing import Any, Dict, Tuple

import torch

from engines.moss_soundeffect_v2.downloader import MossSoundEffectV2Downloader
from utils.audio.cache import get_audio_cache
from utils.models.factory_config import ModelLoadConfig
from utils.models.unified_model_interface import unified_model_interface


class MossSoundEffectV2Adapter:
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.audio_cache = get_audio_cache()
        self._load_config = None

    def _load(self):
        model = self.config.get("model", MossSoundEffectV2Downloader.MODEL_NAME)
        model_path = MossSoundEffectV2Downloader().resolve_model_path(model)
        self._load_config = ModelLoadConfig(
            engine_name="moss_soundeffect_v2",
            model_type="tts",
            model_name=str(model).removeprefix("local:"),
            model_path=model_path,
            device=self.config.get("device", "auto"),
            runtime_mode="main_environment",
            additional_params={"dtype": self.config.get("dtype", "auto")},
        )
        return unified_model_interface.load_model(self._load_config)

    def generate(
        self,
        description: str,
        duration_seconds: float,
        seed: int,
        enable_audio_cache: bool,
    ) -> Tuple[torch.Tensor, int, bool]:
        duration_seconds = round(float(duration_seconds), 1)
        if duration_seconds > 30.0:
            raise ValueError(
                "MOSS-SoundEffect v2 supports a maximum duration of 30 seconds. "
                "Lower duration_seconds in the 🌩️ Sound Effects node."
            )
        params = {
            "description": description,
            "model": self.config.get("model"),
            "duration_seconds": duration_seconds,
            "inference_steps": self.config.get("inference_steps", 100),
            "cfg_scale": self.config.get("cfg_scale", 4.0),
            "sigma_shift": self.config.get("sigma_shift", 5.0),
            "negative_prompt": self.config.get("negative_prompt", ""),
            "seed": int(seed),
            "dtype": self.config.get("dtype", "auto"),
            "device": self.config.get("device", "auto"),
        }
        cache_key = self.audio_cache.generate_cache_key("moss_soundeffect_v2", **params)
        if enable_audio_cache:
            cached = self.audio_cache.get_cached_audio(cache_key)
            if cached is not None:
                return cached[0], 48000, True

        engine = self._load()
        waveform, sample_rate = engine.generate_sound_effect(**params)
        waveform = waveform.detach().cpu().float()
        if enable_audio_cache:
            self.audio_cache.cache_audio(cache_key, waveform, waveform.shape[-1] / float(sample_rate))
        return waveform, int(sample_rate), False
