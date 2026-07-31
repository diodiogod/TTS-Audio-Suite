"""Audio8 TTS Preview engine configuration node."""

import importlib.util
import os
import sys

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


class Audio8TTSEngineNode(BaseTTSNode):
    """Configure the official Audio8 TTS Preview checkpoint."""

    DEFAULT_MODEL = "Audio8-TTS-Preview-0.6b"

    @classmethod
    def NAME(cls):
        return "⚙️ Audio8 TTS Engine"

    @classmethod
    def _get_model_options(cls):
        try:
            from engines.audio8_tts.downloader import Audio8TTSDownloader

            options = Audio8TTSDownloader().get_available_models()
            return options or [cls.DEFAULT_MODEL]
        except Exception:
            return [cls.DEFAULT_MODEL]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (
                    cls._get_model_options(),
                    {
                        "default": cls.DEFAULT_MODEL,
                        "tooltip": (
                            "Official 0.6B Audio8 TTS Preview checkpoint. "
                            "The canonical option downloads from "
                            "Audio8/Audio8-TTS-Preview-0.6b; local: options are "
                            "detected model folders.\n\n"
                            "Preview scope: multilingual speech and zero-shot "
                            "voice cloning, with 11 recommended languages. The "
                            "model infers language from text and has no language "
                            "selection control.\n\n"
                            "Voice cloning requires reference audio and its exact "
                            "matching transcript. Generation without a reference "
                            "is supported. Audio8 has no instruction-conditioned "
                            "voice-design mode. The suite runs it in the existing "
                            "shared Transformers 4 runtime for correct cloning. "
                            "Code and weights: Apache License 2.0."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Execution device. CUDA is recommended. CPU inference "
                            "is supported but slow and runs in float32."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Official sampling temperature. Used only in Sampling "
                            "mode; lower values are more conservative."
                        ),
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Official nucleus-sampling cutoff. Used only in "
                            "Sampling mode."
                        ),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": (
                            "Official top-k sampling limit. Used only in Sampling mode."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": (
                            "Maximum acoustic frames for the first generation "
                            "attempt. Audio8 emits about 21.5 frames per second. "
                            "Larger budgets take more time and context memory."
                        ),
                    },
                ),
            },
            "optional": {
                "retry_max_new_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 64,
                        "max": 2048,
                        "step": 8,
                        "tooltip": (
                            "Second-attempt frame budget when the first attempt "
                            "does not emit EOS. Must be at least max_new_tokens. "
                            "Set it equal to max_new_tokens to disable the larger "
                            "retry."
                        ),
                    },
                ),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Model precision. Auto prefers bfloat16 on Ampere-or-"
                            "newer CUDA GPUs, uses float16 on older CUDA GPUs, and "
                            "float32 on CPU. CPU coerces reduced-precision choices "
                            "to float32."
                        ),
                    },
                ),
                "sampling_mode": (
                    ["Sampling", "Greedy"],
                    {
                        "default": "Sampling",
                        "tooltip": (
                            "Official decoding mode. Sampling uses temperature, "
                            "top_p, and top_k. Greedy is deterministic for a fixed "
                            "prompt and ignores those sampling controls."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_variant: str,
        device: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        retry_max_new_tokens: int = 2048,
        dtype: str = "auto",
        sampling_mode: str = "Sampling",
    ) -> tuple:
        if retry_max_new_tokens < max_new_tokens:
            raise ValueError(
                "Audio8 retry_max_new_tokens must be at least max_new_tokens"
            )

        do_sample = sampling_mode == "Sampling"
        config = {
            "engine_type": "audio8_tts",
            "model_variant": model_variant,
            "device": device,
            "dtype": dtype,
            "max_new_tokens": int(max_new_tokens),
            "retry_max_new_tokens": int(retry_max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "do_sample": do_sample,
        }

        print(f"⚙️ Audio8 TTS Preview: {model_variant} on {device} ({dtype})")
        print(
            "   Settings: "
            f"mode={sampling_mode}, temperature={temperature}, top_p={top_p}, "
            f"top_k={top_k}, max_new_tokens={max_new_tokens}, "
            f"retry_max_new_tokens={retry_max_new_tokens}"
        )
        print(
            "   Voice cloning requires reference audio plus its exact matching "
            "transcript; no-reference generation is supported."
        )
        print(
            "   Runtime: shared Transformers 4 profile (required for correct cloning)"
        )
        print("   Preview checkpoint | Apache License 2.0")

        return (
            {
                "engine_type": "audio8_tts",
                "config": config,
                "adapter_class": "Audio8TTSEngineAdapter",
                "capabilities": ["tts"],
            },
        )
