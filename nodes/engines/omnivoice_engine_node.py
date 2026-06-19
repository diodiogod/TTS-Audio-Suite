"""
OmniVoice engine configuration node.
"""

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


class OmniVoiceEngineNode(BaseTTSNode):
    """OmniVoice engine configuration node."""

    @classmethod
    def NAME(cls):
        return "⚙️ OmniVoice Engine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (["OmniVoice"], {
                    "default": "OmniVoice",
                    "tooltip": "OmniVoice multilingual zero-shot TTS model from k2-fsa/OmniVoice. Voice cloning requires reference audio plus reference text. In this suite, use Character Voices or a narrator voice file with matching .reference.txt. Direct audio-only input is not supported for OmniVoice cloning."
                }),
                "device": (["auto", "cuda", "cpu", "mps", "xpu"], {
                    "default": "auto",
                    "tooltip": "Device to run OmniVoice on. Auto follows the best available backend."
                }),
                "language": ("STRING", {
                    "default": "Auto",
                    "multiline": False,
                    "tooltip": "Target language. Use 'Auto' for language-agnostic generation, or enter a language name/code such as 'English', 'Chinese', 'Japanese', 'en', 'zh', 'ja'. OmniVoice supports 600+ languages."
                }),
                "num_step": ("INT", {
                    "default": 32, "min": 4, "max": 128, "step": 1,
                    "tooltip": "Iterative unmasking steps. Recommended: 32. Use 16 for faster generation."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Recommended: 2.0."
                }),
                "t_shift": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Time-step shift for the OmniVoice noise schedule. Recommended: 0.1."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 3.0, "step": 0.05,
                    "tooltip": "Speech speed factor. Values above 1 are faster; values below 1 are slower. Recommended: 1.0. Technically, this rescales OmniVoice's estimated target audio-token length before generation. It is not waveform time-stretch post-processing."
                }),
                "duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Fixed output duration in seconds. Set 0 to let OmniVoice estimate duration from text. Technically, this overrides speed and sets the planned target audio-token length from seconds, so it behaves more like native target-length control than a simple max-tokens cap."
                }),
            },
            "optional": {
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Runtime precision. Auto prefers bf16 on newer CUDA/XPU hardware, else fp16, with fp32 on CPU."
                }),
                "instruct": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional OmniVoice voice-design / style instruction. Works for pure voice design (no reference audio) or as extra guidance with a properly transcribed reference voice."
                }),
                "layer_penalty_factor": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Penalty encouraging earlier codebook layers to unmask first. Recommended: 5.0."
                }),
                "position_temperature": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Mask-position temperature. 0 is greedy; higher values add randomness. Recommended: 5.0."
                }),
                "class_temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Token sampling temperature. 0 is greedy; higher values are more random. Recommended: 0.0."
                }),
                "denoise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable the denoise token. Recommended: on."
                }),
                "preprocess_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preprocess the voice-cloning reference by trimming long reference audio, removing silence, and adding terminal punctuation to the reference text. Recommended: on."
                }),
                "postprocess_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Post-process generated audio with silence cleanup plus fade and edge padding. Recommended: on."
                }),
                "audio_chunk_duration": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Native long-form target chunk duration in seconds. This is OmniVoice's real long-text control and supersedes suite char-based chunking. Recommended: 15.0."
                }),
                "audio_chunk_threshold": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 180.0, "step": 0.5,
                    "tooltip": "Native long-form activation threshold in seconds. Estimated outputs longer than this use OmniVoice's built-in chunking path. Recommended: 30.0."
                }),
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
        language: str,
        num_step: int,
        guidance_scale: float,
        t_shift: float,
        speed: float,
        duration: float,
        dtype: str = "auto",
        instruct: str = "",
        layer_penalty_factor: float = 5.0,
        position_temperature: float = 5.0,
        class_temperature: float = 0.0,
        denoise: bool = True,
        preprocess_prompt: bool = True,
        postprocess_output: bool = True,
        audio_chunk_duration: float = 15.0,
        audio_chunk_threshold: float = 30.0,
    ) -> tuple:
        config = {
            "engine_type": "omnivoice",
            "model_variant": model_variant,
            "device": device,
            "dtype": dtype,
            "language": str(language or "Auto").strip() or "Auto",
            "instruct": str(instruct or "").strip(),
            "num_steps": int(num_step),
            "guidance_scale": float(guidance_scale),
            "t_shift": float(t_shift),
            "speed": float(speed),
            "duration": float(duration),
            "layer_penalty_factor": float(layer_penalty_factor),
            "position_temperature": float(position_temperature),
            "class_temperature": float(class_temperature),
            "denoise": bool(denoise),
            "preprocess_prompt": bool(preprocess_prompt),
            "postprocess_output": bool(postprocess_output),
            "audio_chunk_duration": float(audio_chunk_duration),
            "audio_chunk_threshold": float(audio_chunk_threshold),
        }

        print(f"⚙️ OmniVoice: Configured on {device}")
        print(
            f"   Model: {model_variant} | Language: {config['language']} | Dtype: {dtype}"
        )
        print(
            "   Settings: steps={steps}, guidance_scale={guidance}, t_shift={t_shift}, speed={speed}, duration={duration}".format(
                steps=num_step,
                guidance=guidance_scale,
                t_shift=t_shift,
                speed=speed,
                duration="auto" if duration <= 0 else duration,
            )
        )
        print(
            "   Native long-form: chunk_duration={chunk_duration}s, threshold={threshold}s".format(
                chunk_duration=audio_chunk_duration,
                threshold=audio_chunk_threshold,
            )
        )
        print("   Voice cloning requires reference text. Use Character Voices or a .reference.txt narrator voice.")
        if instruct:
            print(f"   Instruct: {instruct[:80]}..." if len(instruct) > 80 else f"   Instruct: {instruct}")

        engine_data = {
            "engine_type": "omnivoice",
            "config": config,
            "adapter_class": "OmniVoiceEngineAdapter",
        }
        return (engine_data,)
