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
                    "tooltip": "Target language. OmniVoice supports 600+ languages. Auto uses language-agnostic generation; an explicit name/code can improve pronunciation and conditioning, especially for short or ambiguous text. Examples: English, Chinese, Japanese, en, zh, ja."
                }),
                "num_step": ("INT", {
                    "default": 32, "min": 4, "max": 128, "step": 1,
                    "tooltip": "Iterative decoding steps. More steps can improve convergence, clarity, and difficult generations, but increase render time with diminishing returns. 32 is the quality default; 16 is a faster preview setting."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Strength of text, language, and instruction conditioning. Higher values can follow conditioning more strongly; excessive guidance may sound forced, distorted, or less natural. Recommended starting point: 2.0."
                }),
                "t_shift": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Shapes how decoding work is distributed across the noise schedule. It can affect convergence and detail, but has no simple quality direction. Keep the tuned 0.1 default unless diagnosing a specific generation problem."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 3.0, "step": 0.05,
                    "tooltip": "Native speech-rate control. Above 1 generates fewer audio tokens for faster speech; below 1 generates more for slower speech. Extreme values may reduce naturalness. This changes generation length, not waveform playback speed. Ignored when duration is set. Recommended: 1.0."
                }),
                "duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Fixed generated length in seconds. 0 lets OmniVoice estimate naturally. A positive value overrides speed and plans the audio-token length directly; unrealistic durations can cause rushed, stretched, or unstable speech. Output cleanup may trim trailing silence."
                }),
            },
            "optional": {
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision. Auto chooses an appropriate format. BF16 is usually the safest reduced precision on supported GPUs; FP16 may be faster on some hardware; FP32 uses much more memory and is mainly useful for compatibility diagnosis."
                }),
                "instruct": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "In Text to Speech mode, optionally guides speaker attributes such as gender, age, pitch, whisper, supported English accents, or Chinese dialects. A reference voice takes priority. In Voice Design mode this field is disabled because Voice Designer supplies the design instruction."
                }),
                "layer_penalty_factor": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Controls how strongly earlier audio-codebook layers are resolved before deeper detail layers. The tuned value 5.0 prioritizes coarse speech structure first; unusual values can disrupt decoding quality. Usually leave unchanged."
                }),
                "position_temperature": ("FLOAT", {
                    "default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Randomness in which masked audio positions are filled next. Lower values are more repeatable and may reduce variation; 0 uses greedy position selection. Higher values increase diversity but can make results less consistent. Default: 5.0."
                }),
                "class_temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Randomness when selecting audio tokens. 0 uses greedy token selection for maximum consistency. Raising it can add variation, but also increases the chance of artifacts or unstable speech. Default: 0.0."
                }),
                "denoise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Requests cleaner speech through OmniVoice's denoise token when reference audio is used. It does not affect the current reference-free path. Recommended: on."
                }),
                "preprocess_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cleans voice-cloning input by trimming overly long reference audio and silence, and normalizing terminal punctuation in its transcript. Helps produce a compact, reliable prompt. Has no effect without reference audio. Recommended: on."
                }),
                "postprocess_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cleans generated audio by removing long silence and applying edge fades/padding. Disable only when raw model output or exact requested duration matters, since silence cleanup may shorten the result slightly. Recommended: on."
                }),
                "audio_chunk_duration": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Target size of OmniVoice's native long-form chunks. Smaller chunks reduce VRAM and can stabilize difficult long text, but create more boundaries; larger chunks preserve more context but cost more memory. Default: 15 seconds."
                }),
                "audio_chunk_threshold": ("FLOAT", {
                    "default": 30.0, "min": 1.0, "max": 180.0, "step": 0.5,
                    "tooltip": "Estimated output duration above which native long-form chunking activates. Lower it to chunk shorter passages for stability/VRAM; raise it to keep more text in one generation. Default: 30 seconds."
                }),
                "mode": (["Text to Speech", "Voice Design"], {
                    "default": "Text to Speech",
                    "tooltip": (
                        "Restricts this engine instance to one operation. Text to Speech works with TTS Text/SRT "
                        "and enables this engine's instruction. Voice Design works only with Voice Designer, "
                        "which supplies the instruction. Duplicate the engine node if a workflow needs both modes."
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
        mode: str = "Text to Speech",
    ) -> tuple:
        model_role = "voice_design" if mode == "Voice Design" else "tts"
        config = {
            "engine_type": "omnivoice",
            "model_variant": model_variant,
            "mode": mode,
            "model_role": model_role,
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

        print(f"⚙️ OmniVoice: Configured for {mode} on {device}")
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
            "capabilities": [model_role],
        }
        return (engine_data,)
