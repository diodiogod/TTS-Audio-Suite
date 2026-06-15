"""
Unified Sound Effects Node - engine-agnostic text-to-sound generation.
"""

import hashlib
import importlib.util
import os
import sys
import time
from typing import Any, Dict, Tuple

import torch

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseTTSNode = base_module.BaseTTSNode

from engines.adapters import DotsTTSEngineAdapter


class UnifiedSoundEffectsNode(BaseTTSNode):
    """Unified text-to-sound node for engines that support non-speech audio generation."""

    ENGINE_ADAPTERS = {
        "dots_tts": DotsTTSEngineAdapter,
    }

    TEMPLATE_OVERRIDES = {
        "dots_tts": "text_to_audio",
    }

    @classmethod
    def NAME(cls):
        return "🎧 Sound Effects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": ("TTS_ENGINE", {
                    "tooltip": "Engine configuration for a backend that supports sound-effects generation. Currently Dots text_to_audio is supported."
                }),
                "description": ("STRING", {
                    "multiline": True,
                    "default": "Heavy rain on a metal rooftop with distant thunder and occasional wind gusts.",
                    "tooltip": "Describe the non-speech audio you want. This node is for sound-effects / scene audio generation, not spoken TTS."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible generation. Set 0 for random output."
                }),
            },
            "optional": {
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache generated sound effects in memory so identical reruns are instant."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_sound"
    CATEGORY = "TTS Audio Suite/🎧 Sound Effects"

    def __init__(self):
        super().__init__()
        self._cached_adapters = {}

    def _validate_engine(self, engine: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(engine, dict):
            raise ValueError("Engine input must be a configuration dict")

        capabilities = engine.get("capabilities", [])
        if capabilities and "sound_effects" not in capabilities:
            raise ValueError("Engine does not support Sound Effects")

        config = engine.get("config", engine)
        engine_type = engine.get("engine_type") or config.get("engine_type")
        if engine_type not in self.ENGINE_ADAPTERS:
            raise ValueError(
                f"Engine '{engine_type}' is not wired for Sound Effects yet. "
                "This family exists now, but only Dots text_to_audio is implemented."
            )
        return config.copy()

    def _get_adapter(self, engine_type: str, config: Dict[str, Any]):
        stable_params = {
            "engine_type": engine_type,
            "model_variant": config.get("model_variant"),
            "device": config.get("device"),
            "precision": config.get("precision"),
            "optimize": config.get("optimize"),
            "max_generate_length": config.get("max_generate_length"),
            "template_name": config.get("template_name"),
        }
        cache_key = hashlib.md5(str(sorted(stable_params.items())).encode()).hexdigest()[:12]
        adapter = self._cached_adapters.get(cache_key)
        if adapter is None:
            adapter_cls = self.ENGINE_ADAPTERS[engine_type]
            adapter = adapter_cls(config=config)
            self._cached_adapters[cache_key] = adapter
        else:
            adapter.update_config(config)
        return adapter

    def _generate_with_dots(
        self,
        config: Dict[str, Any],
        description: str,
        seed: int,
        enable_audio_cache: bool,
    ):
        config["template_name"] = self.TEMPLATE_OVERRIDES["dots_tts"]
        adapter = self._get_adapter("dots_tts", config)
        return adapter.generate_single(
            text=description,
            voice_ref=None,
            seed=seed,
            enable_audio_cache=enable_audio_cache,
            character_name="sound_effects",
        )

    def generate_sound(
        self,
        engine: Dict[str, Any],
        description: str,
        seed: int,
        enable_audio_cache: bool = True,
    ) -> Tuple[Dict[str, Any], str]:
        config = self._validate_engine(engine)
        engine_type = config.get("engine_type", engine.get("engine_type", "unknown"))
        clean_description = (description or "").strip()
        if not clean_description:
            raise ValueError("Sound Effects description cannot be empty")

        print(f"🎧 Sound Effects: Starting {engine_type} generation")
        print("============================================================")
        print(clean_description)
        print("============================================================")

        start_time = time.time()

        if engine_type == "dots_tts":
            audio_tensor = self._generate_with_dots(
                config=config,
                description=clean_description,
                seed=int(seed or 0),
                enable_audio_cache=bool(enable_audio_cache),
            )
        else:
            raise ValueError(f"Unsupported Sound Effects engine: {engine_type}")

        elapsed = time.time() - start_time
        sample_rate = 48000 if engine_type == "dots_tts" else 44100
        audio_duration = 0.0
        if hasattr(audio_tensor, "shape") and audio_tensor.shape[-1] > 0:
            audio_duration = float(audio_tensor.shape[-1]) / float(sample_rate)

        print(
            f"✅ Sound Effects: {engine_type} generation complete "
            f"({audio_duration:.2f}s audio in {elapsed:.1f}s)"
        )

        info = (
            f"Sound Effects generated successfully\n"
            f"Engine: {engine_type}\n"
            f"Mode: {self.TEMPLATE_OVERRIDES.get(engine_type, 'custom')}\n"
            f"Duration: {audio_duration:.2f}s\n"
            f"Elapsed: {elapsed:.2f}s\n"
            f"Seed: {int(seed or 0)}"
        )

        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
        audio_tensor = audio_tensor.detach().float().cpu()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        return ({
            "waveform": audio_tensor,
            "sample_rate": sample_rate,
        }, info)


NODE_CLASS_MAPPINGS = {
    "UnifiedSoundEffectsNode": UnifiedSoundEffectsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedSoundEffectsNode": "🎧 Sound Effects"
}
