"""Higgs Audio v3 engine configuration node."""

import os
import sys
import importlib.util


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)
BaseTTSNode = base_module.BaseTTSNode


class HiggsAudioV3EngineNode(BaseTTSNode):
    """Configure Higgs Audio v3 for unified TTS nodes."""

    @classmethod
    def NAME(cls):
        return "⚙️ Higgs Audio v3 Engine"

    @classmethod
    def _get_available_models(cls):
        try:
            from engines.higgs_audio_v3.higgs_audio_v3_downloader import HiggsAudioV3Downloader

            return HiggsAudioV3Downloader().get_available_models()
        except Exception:
            return ["higgs-audio-v3-tts-4b"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls._get_available_models(), {
                    "default": "higgs-audio-v3-tts-4b",
                    "tooltip": (
                        "Higgs Audio v3 TTS 4B official Boson model. "
                        "Downloads to ComfyUI/models/TTS/higgs_audio_v3/. "
                        "License: research and non-commercial use only."
                    ),
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device for native Higgs Audio v3 inference. CUDA is strongly recommended; CPU is very slow.",
                }),
                "dtype": (["auto", "bf16", "fp32"], {
                    "default": "auto",
                    "tooltip": "auto uses bf16 on supported CUDA and fp32 otherwise. Use fp32 if bf16 produces unstable audio.",
                }),
                "attention": (["auto", "sdpa", "eager", "flash_attention", "sageattention"], {
                    "default": "auto",
                    "tooltip": "Attention backend. auto/sdpa is safest. flash_attention and sageattention require optional packages.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Native sampling temperature. 0 is greedy; 0.8-1.1 is usually natural.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Native nucleus sampling cutoff. 1.0 disables top-p filtering.",
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 0, "max": 1026, "step": 1,
                    "tooltip": "Native top-k codebook sampling. 0 disables top-k filtering.",
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048, "min": 32, "max": 8192, "step": 8,
                    "tooltip": "Maximum generated audio-token steps per call. Raise if speech cuts off; prefer chunking for long text.",
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("tts_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    DESCRIPTION = (
        "Configure Higgs Audio v3 TTS. Type official inline tags directly in text, e.g. "
        "<|emotion:amusement|>, <|style:whispering|>, <|sfx:laughter|>Haha."
    )

    def create_engine_config(
        self,
        model,
        device,
        dtype,
        attention,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
    ):
        config = {
            "engine_type": "higgs_audio_v3",
            "model": model,
            "device": device,
            "dtype": dtype,
            "attention": attention,
            "temperature": max(0.0, min(2.0, float(temperature))),
            "top_p": max(0.0, min(1.0, float(top_p))),
            "top_k": max(0, min(1026, int(top_k))),
            "max_new_tokens": max(32, min(8192, int(max_new_tokens))),
            "adapter_class": "HiggsAudioV3EngineAdapter",
        }
        print(f"✅ Higgs Audio v3 engine config created: {model} on {device}")
        return (config,)


NODE_CLASS_MAPPINGS = {
    "HiggsAudioV3EngineNode": HiggsAudioV3EngineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsAudioV3EngineNode": "⚙️ Higgs Audio v3 Engine",
}
