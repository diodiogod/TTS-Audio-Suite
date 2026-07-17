"""Configuration node for the standalone MOSS-SoundEffect v2 family."""

import os

from engines.moss_soundeffect_v2.downloader import MossSoundEffectV2Downloader
from utils.models.extra_paths import get_all_tts_model_paths


class MossSoundEffectV2EngineNode:
    MODEL_NAME = MossSoundEffectV2Downloader.MODEL_NAME

    @classmethod
    def _model_options(cls):
        options = []
        for base_path in get_all_tts_model_paths("TTS"):
            engine_dir = os.path.join(base_path, "moss_soundeffect_v2")
            if not os.path.isdir(engine_dir):
                continue
            for name in sorted(os.listdir(engine_dir)):
                candidate = os.path.join(engine_dir, name)
                local_option = f"local:{name}"
                if MossSoundEffectV2Downloader.is_complete(candidate) and local_option not in options:
                    options.append(local_option)
        options.append(cls.MODEL_NAME)
        return options

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls._model_options(), {
                    "default": cls._model_options()[0],
                    "tooltip": (
                        "Official MOSS-SoundEffect v2 model. A local: entry means it is already downloaded.\n"
                        "The DiT uses torch.compile automatically. Its first generation can spend several minutes "
                        "compiling. Compatible compiled artifacts are cached and reused across ComfyUI sessions."
                    ),
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Generation device. CUDA is strongly recommended; CPU generation is impractically slow.",
                }),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision. Auto prefers bfloat16 on supported GPUs and float16 otherwise.",
                }),
                "inference_steps": ("INT", {
                    "default": 100, "min": 10, "max": 150, "step": 5,
                    "tooltip": "Diffusion steps. More steps are slower and may improve detail; 100 is the official default.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Prompt guidance strength. Higher values follow the description more strongly but can sound less natural.",
                }),
                "sigma_shift": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Flow-matching schedule shift. The official default is 5; change it only for deliberate experimentation.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Optional sounds or qualities to discourage. Leave empty unless the output repeatedly contains an unwanted element.",
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine(self, model, device, dtype, inference_steps, cfg_scale, sigma_shift, negative_prompt):
        config = {
            "engine_type": "moss_soundeffect_v2",
            "model": model,
            "device": device,
            "dtype": dtype,
            "inference_steps": int(inference_steps),
            "cfg_scale": float(cfg_scale),
            "sigma_shift": float(sigma_shift),
            "negative_prompt": str(negative_prompt or ""),
            "runtime_mode": "main_environment",
        }
        return ({
            "engine_type": "moss_soundeffect_v2",
            "engine_name": "MOSS-SoundEffect v2",
            "config": config,
            "capabilities": ["sound_effects"],
        },)


NODE_CLASS_MAPPINGS = {"MossSoundEffectV2EngineNode": MossSoundEffectV2EngineNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossSoundEffectV2EngineNode": "⚙️ MOSS SoundEffect v2 Engine"}
