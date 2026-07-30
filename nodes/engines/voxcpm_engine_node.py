"""VoxCPM engine configuration node."""

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


class VoxCPMEngineNode(BaseTTSNode):
    """Configure the official VoxCPM runtime for all released generations."""

    DOWNLOADABLE_MODELS = ["VoxCPM2", "VoxCPM1.5", "VoxCPM-0.5B"]
    NO_LORA_OPTION = "None"

    @classmethod
    def _get_lora_options(cls):
        try:
            from engines.voxcpm.training.common import discover_lora_adapters

            return [cls.NO_LORA_OPTION] + discover_lora_adapters()
        except Exception:
            return [cls.NO_LORA_OPTION]

    @classmethod
    def NAME(cls):
        return "⚙️ VoxCPM Engine"

    @classmethod
    def _get_model_options(cls):
        try:
            from engines.voxcpm.voxcpm_downloader import VoxCPMDownloader

            downloader = VoxCPMDownloader()
            if hasattr(downloader, "get_available_models"):
                return downloader.get_available_models()
        except Exception:
            pass
        return list(cls.DOWNLOADABLE_MODELS)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (cls._get_model_options(), {
                    "default": "VoxCPM2",
                    "tooltip": (
                        "Official OpenBMB checkpoint.\n"
                        "• VoxCPM2: recommended; about 8 GB VRAM, 30 languages, "
                        "reference-only cloning, controllable cloning, and Voice Design\n"
                        "• VoxCPM1.5: about 6 GB VRAM, faster, English/Chinese cloning\n"
                        "• VoxCPM-0.5B: legacy 16 kHz model; kept for lowest-memory compatibility\n"
                        "Options prefixed with local: are complete checkpoints already on disk."
                    ),
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": (
                        "Inference device. Auto selects the best available backend. "
                        "CPU is very slow. The official runtime forces float32 on MPS "
                        "because half precision degrades audio."
                    ),
                }),
                "cfg_value": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": (
                        "Official classifier-free guidance value. OpenBMB uses 2.0. "
                        "Higher is not automatically better and can make speech less natural."
                    ),
                }),
                "inference_timesteps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "Official diffusion sampling steps. 10 is the quality/speed default; "
                        "more steps cost proportionally more time."
                    ),
                }),
                "max_len": ("INT", {
                    "default": 4096,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "tooltip": (
                        "Maximum generated audio-token length per suite chunk. VoxCPM2 and "
                        "1.5 support up to 8192; the legacy 0.5B checkpoint supports 4096."
                    ),
                }),
            },
            "optional": {
                "local_lora_adapter": (cls._get_lora_options(), {
                    "default": cls.NO_LORA_OPTION,
                    "tooltip": (
                        "Optional VoxCPM2 LoRA under models/TTS/voxcpm/loras. "
                        "Legacy VoxCPM models do not accept these adapters."
                    ),
                }),
                "lora_adapter_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional absolute VoxCPM LoRA folder path. Overrides the dropdown.",
                }),
                "voice_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "VoxCPM2 only. Describe age, timbre, accent, emotion, or delivery. "
                        "Without reference audio it designs a voice; with reference audio it "
                        "performs controllable cloning. Legacy models reject this input."
                    ),
                }),
                "normalize_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run the official VoxCPM text normalizer before inference.",
                }),
                "retry_badcase": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Retry generations whose audio-to-text length ratio suggests a bad case. "
                        "This is an official VoxCPM safeguard and can make a render take longer."
                    ),
                }),
                "retry_badcase_max_times": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Maximum official bad-case generation attempts.",
                }),
                "retry_badcase_ratio_threshold": ("FLOAT", {
                    "default": 6.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Official audio-to-text length-ratio threshold used for bad-case retries.",
                }),
                "optimize": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Enable the official torch.compile optimization on CUDA. First load and "
                        "generation become slower; leave disabled for CPU, MPS, and easier unloading."
                    ),
                }),
                "mode": (["Text to Speech", "Voice Design"], {
                    "default": "Text to Speech",
                    "tooltip": (
                        "Text to Speech works with unified TTS Text/SRT. Voice Design is supported "
                        "only by VoxCPM2 and works with the Voice Designer node. Duplicate this engine "
                        "node if one workflow needs both roles."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_variant,
        device,
        cfg_value,
        inference_timesteps,
        max_len,
        local_lora_adapter="None",
        lora_adapter_override="",
        voice_instruction="",
        normalize_text=False,
        retry_badcase=True,
        retry_badcase_max_times=3,
        retry_badcase_ratio_threshold=6.0,
        optimize=False,
        mode="Text to Speech",
    ):
        selected_lora = str(lora_adapter_override or "").strip()
        if not selected_lora and str(local_lora_adapter) != self.NO_LORA_OPTION:
            selected_lora = str(local_lora_adapter).strip()
        if selected_lora:
            selected_lora = os.path.abspath(os.path.expanduser(selected_lora))
            from engines.voxcpm.training.common import read_lora_info

            read_lora_info(selected_lora)
            if model_variant in {"VoxCPM1.5", "VoxCPM-0.5B"}:
                raise ValueError("VoxCPM LoRA adapters currently require VoxCPM2")

        if model_variant == "VoxCPM-0.5B" and int(max_len) > 4096:
            max_len = 4096

        model_role = "voice_design" if mode == "Voice Design" else "tts"
        if model_role == "voice_design" and model_variant in {"VoxCPM1.5", "VoxCPM-0.5B"}:
            raise ValueError(
                f"{model_variant} cannot design voices. Select VoxCPM2 or a local VoxCPM2 checkpoint."
            )

        config = {
            "engine_type": "voxcpm",
            "model_variant": str(model_variant),
            "model_name": str(model_variant),
            "model_role": model_role,
            "mode": str(mode),
            "device": str(device),
            "cfg_value": float(cfg_value),
            "inference_timesteps": int(inference_timesteps),
            "min_len": 2,
            "max_len": int(max_len),
            "voice_instruction": str(voice_instruction or "").strip(),
            "normalize_text": bool(normalize_text),
            "retry_badcase": bool(retry_badcase),
            "retry_badcase_max_times": int(retry_badcase_max_times),
            "retry_badcase_ratio_threshold": float(retry_badcase_ratio_threshold),
            "optimize": bool(optimize),
            "runtime_mode": "main_environment",
            "lora_adapter": selected_lora or None,
        }

        print(
            f"⚙️ VoxCPM: {model_variant} on {device} | {mode} | "
            f"cfg={float(cfg_value):g}, steps={int(inference_timesteps)}, max_len={int(max_len)}"
        )
        if selected_lora:
            print(f"   LoRA adapter: {selected_lora}")
        return ({
            "engine_type": "voxcpm",
            "config": config,
            "adapter_class": "VoxCPMEngineAdapter",
            "capabilities": [model_role],
        },)
