"""
Qwen3-TTS Engine Configuration Node

Provides one explicit model selector for Qwen3-TTS Base, CustomVoice, and
VoiceDesign checkpoints.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

import folder_paths
from engines.qwen3_asr.prompting import DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE
from engines.qwen3_tts.qwen3_tts_downloader import Qwen3TTSDownloader
from utils.models.extra_paths import get_all_tts_model_paths
from utils.models.factory_config import (
    RUNTIME_MODE_DEDICATED,
    RUNTIME_MODE_MAIN,
    RUNTIME_MODE_SHARED,
    normalize_runtime_mode,
)


RUNTIME_MODE_MAIN_LABEL = "Main Environment"
RUNTIME_MODE_SHARED_LABEL = "⚠️ Shared Runtime"
RUNTIME_MODE_DEDICATED_LABEL = "⚠️ Dedicated Runtime"


class Qwen3TTSEngineNode(BaseTTSNode):
    """
    Qwen3-TTS Engine configuration node.
    Unified interface for all 3 model variants (CustomVoice/VoiceDesign/Base).
    """

    LEGACY_MODEL_VALUES = ("1.7B", "0.6B")

    @classmethod
    def _model_specs(cls):
        return {
            name: spec
            for name, spec in Qwen3TTSDownloader.MODELS.items()
            if spec.get("role") in {"tts", "voice_design"}
        }

    @classmethod
    def _find_local_model(cls, model_name: str) -> str:
        try:
            for base_path in get_all_tts_model_paths("TTS"):
                candidates = (
                    os.path.join(base_path, "qwen3_tts", model_name),
                    os.path.join(base_path, "Qwen3-TTS", model_name),
                    os.path.join(base_path, model_name),
                )
                for candidate in candidates:
                    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                        return f"local:{model_name}"
        except Exception:
            pass
        return ""

    @classmethod
    def _model_options(cls) -> List[str]:
        preferred_order = (
            "Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen3-TTS-12Hz-0.6B-Base",
            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        )
        specs = cls._model_specs()
        options = []
        for model_name in preferred_order:
            spec = specs[model_name]
            options.append(cls._find_local_model(model_name) or spec["display"])
        # Kept at the end so old API/UI workflows validate long enough to be normalized.
        options.extend(cls.LEGACY_MODEL_VALUES)
        return options

    @classmethod
    def _resolve_model(cls, selected: str, voice_preset: str) -> tuple[str, str, Dict[str, Any]]:
        specs = cls._model_specs()
        selected = str(selected or "").strip()

        if selected in cls.LEGACY_MODEL_VALUES:
            model_type = (
                "Base"
                if voice_preset == "None (Zero-shot / Custom)"
                else "CustomVoice"
            )
            model_name = f"Qwen3-TTS-12Hz-{selected}-{model_type}"
            local = cls._find_local_model(model_name)
            resolved = local or model_name
            return resolved, model_name, specs[model_name]

        display_to_name = {spec["display"]: name for name, spec in specs.items()}
        if selected in display_to_name:
            model_name = display_to_name[selected]
            return cls._find_local_model(model_name) or model_name, model_name, specs[model_name]

        model_name = selected.removeprefix("local:")
        if model_name not in specs:
            available = ", ".join(spec["display"] for spec in specs.values())
            raise ValueError(f"Unknown Qwen3-TTS model '{selected}'. Available models: {available}")
        return selected, model_name, specs[model_name]

    @classmethod
    def NAME(cls):
        return "⚙️ Qwen3-TTS Engine"

    @classmethod
    def INPUT_TYPES(cls):
        # Preset voices for CustomVoice model
        voice_presets = [
            "None (Zero-shot / Custom)",
            "Vivian",
            "Serena",
            "Uncle_Fu",
            "Dylan",
            "Eric",
            "Ryan",
            "Aiden",
            "Ono_Anna",
            "Sohee"
        ]
        asr_translate_languages = [
            "English",
            "Chinese",
            "Cantonese",
            "Arabic",
            "German",
            "French",
            "Spanish",
            "Portuguese",
            "Indonesian",
            "Italian",
            "Korean",
            "Russian",
            "Thai",
            "Vietnamese",
            "Japanese",
            "Turkish",
            "Hindi",
            "Malay",
            "Dutch",
            "Swedish",
            "Danish",
            "Finnish",
            "Polish",
            "Czech",
            "Filipino",
            "Persian",
            "Greek",
            "Romanian",
            "Hungarian",
            "Macedonian",
        ]

        return {
            "required": {
                # Model Configuration
                "model_variant": (cls._model_options(), {
                    "default": cls._model_options()[0],
                    "tooltip": "Explicit Qwen3-TTS checkpoint. Local installations use the local: prefix. Base models clone a reference voice, CustomVoice models use preset speakers, and VoiceDesign works only with Unified Voice Designer."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run Qwen3-TTS model on:\n• auto: Automatically select best available\n• cuda: NVIDIA GPU (requires CUDA)\n• cpu: CPU-only processing (very slow)"
                }),

                # Voice Control
                "voice_preset": (voice_presets, {
                    "default": "None (Zero-shot / Custom)",
                    "tooltip": "Preset speaker for CustomVoice checkpoints. Ignored by Base and VoiceDesign models."
                }),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {
                    "default": "Auto",
                    "tooltip": "Target language for speech generation:\n• Auto: Automatically detect from text\n• Portuguese: European Portuguese only for Base model\n  - Brazilian Portuguese: Only available with CustomVoice presets + instruction\n  - Example instruction: 'Speak with Brazilian Portuguese accent'"
                }),
                "instruct": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Voice description for VoiceDesign, or delivery instruction for the 1.7B CustomVoice checkpoint. Base and 0.6B CustomVoice models ignore it."
                }),

                # Generation Parameters
                "top_k": ("INT", {
                    "default": 50, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Top-K sampling:\n• Lower (10-30): More focused, consistent\n• Default (50): Balanced\n• Higher (80-100): More varied output"
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Top-P (nucleus) sampling:\n• Lower (0.7-0.9): More focused\n• Default (1.0): All tokens considered\n• Combine with top_k for fine control"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Sampling temperature:\n• Lower (0.5-0.7): More predictable\n• Default (0.9): Balanced\n• Higher (1.2+): More creative, potentially unstable"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Repetition penalty:\n• 1.0: No penalty\n• 1.05: Slight penalty (default)\n• 1.2+: Strong penalty (may affect quality)"
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048, "min": 64, "max": 8192, "step": 64,
                    "tooltip": "Maximum tokens to generate:\n• 64-512: Very short\n• 2048: Standard (default)\n• 4096-8192: Long generation\nHigher = more VRAM + longer generation"
                }),
            },
            "optional": {
                # Advanced Parameters
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision:\n• auto: Selects bfloat16 if GPU supports it (SM 8.0+), else float16\n• bfloat16: Best quality, stable (RTX 30xx+, A100+)\n• float16: Good quality, wider compatibility\n• float32: Maximum precision, 2x VRAM"
                }),
                "attn_implementation": (["auto", "sage_attn", "flash_attention_2", "sdpa", "eager"], {
                    "default": "auto",
                    "tooltip": "Attention mechanism:\n• auto: Best available (priority: sage_attn > flash_attention_2 > sdpa > eager)\n• sage_attn: Fastest (requires sageattention package)\n• flash_attention_2: Very fast (requires flash-attn installed)\n• sdpa: Good balance (PyTorch native)\n• eager: Slowest, most compatible"
                }),
                "x_vector_only_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "X-vector only mode (Base model only):\n• False: Full voice cloning (high quality, requires ref_text)\n• True: Fast speaker embedding only (lower quality, no ref_text needed)\nOnly applies when voice_preset = None"
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable torch.compile for decoder (~1.5-2x speedup):\n• False: Standard inference (RECOMMENDED - works with all PyTorch versions)\n• True: Compiled decoder (REQUIRES PyTorch 2.10+ and triton-windows 3.6+)\n⚠️ REQUIREMENTS: PyTorch 2.10.0+cu130, triton-windows 3.6.0+, and Visual Studio C++ Build Tools on Windows\n⚠️ Shared/Dedicated Runtime will try to detect the toolchain automatically, but it still must be installed\n⚠️ See docs/qwen3_tts_optimizations.md for installation\nFirst generation slower due to compilation, then ~1.5-2x faster"
                }),
                "use_cuda_graphs": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable manual CUDA graph capture:\n• False: RECOMMENDED (reduce-overhead mode already has auto CUDA graphs)\n• True: Manual capture - minimal gain (~0.1 it/s), only works with mode='default'\nNote: reduce-overhead/max-autotune include CUDA graphs automatically"
                }),
                "compile_mode": (["default", "reduce-overhead", "max-autotune"], {
                    "default": "default",
                    "tooltip": "torch.compile mode (if use_torch_compile enabled):\n• default: RECOMMENDED - Standard torch.compile, ~1.7x speedup, works on Windows\n• reduce-overhead: Auto CUDA graphs, ~2-3x speedup, LINUX ONLY (fails on Windows)\n• max-autotune: Best optimization, longest compile, LINUX ONLY\n⚠️ Windows: Only 'default' works - reduce-overhead/max-autotune fail with cudaMallocAsync error\n⚠️ Model must be reloaded to test different modes (cache reuse otherwise)"
                }),
                "asr_use_forced_aligner": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Qwen3 forced aligner (required for word timestamps + accurate SRT).\nTurn OFF to save VRAM and skip downloading the extra aligner model if you only need text (no timestamps)."
                }),
                "asr_translate_target_language": (asr_translate_languages, {
                    "default": "English",
                    "tooltip": "Experimental ASR translation target for ✏️ ASR Transcribe when task=translate.\n\nThis is ASR-only and does NOT affect Qwen TTS generation language.\n\nImportant:\n• Qwen translation in this integration is prompt-driven through the ASR wrapper, not a dedicated native target-language API\n• Translation quality currently depends heavily on language pair and backend behavior\n• English is the safest default\n• Non-English targets should be treated as experimental until you validate them on real audio"
                }),
                "asr_translate_instruction_override": ("STRING", {
                    "default": DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE,
                    "multiline": True,
                    "tooltip": "Qwen ASR-only experimental translation instruction template for ✏️ ASR Transcribe when task=translate.\n\nThis field shows the actual default instruction used by this integration. Edit it if you want to experiment.\n\nAvailable placeholders:\n• {source_language}: Replaced with the unified ASR source language, or 'the spoken source language' when ASR language is Auto\n• {target_language}: Replaced with this engine node's ASR translation target\n\nImportant:\n• This does NOT affect Qwen TTS generation or the TTS 'instruct' field\n• Qwen translation here is prompt-driven through the ASR wrapper context\n• Results can vary a lot by language pair, and some pairs may not behave reliably at all\n• Weak or malformed custom instructions can produce worse translations or unexpected output"
                }),
                "runtime_mode": ([RUNTIME_MODE_MAIN_LABEL, RUNTIME_MODE_SHARED_LABEL, RUNTIME_MODE_DEDICATED_LABEL], {
                    "default": RUNTIME_MODE_SHARED_LABEL,
                    "tooltip": "IMPORTANT: Qwen3-TTS is fragile on Transformers 5 in the main environment.\n\nRuntime Isolation:\n• Main Environment: Use the main ComfyUI Python environment\n• Shared Runtime: Use the shared secondary legacy runtime already used by compatible engines\n• Dedicated Runtime: Create a separate secondary runtime just for Qwen3-TTS\n\nWhy this matters:\n• The main ComfyUI env is on Transformers 5\n• Qwen3-TTS is more stable on the legacy Transformers 4 stack\n• Runtime isolation keeps Qwen3-TTS working without downgrading the whole app\n\n⚠️ Shared/Dedicated runtimes currently reuse heavy base packages from the main env (like PyTorch) and install pinned Qwen3-TTS-specific packages on top.\n⚠️ First run may create the secondary runtime and take a while."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_variant: str,
        device: str,
        voice_preset: str,
        language: str,
        instruct: str,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        runtime_mode: str = RUNTIME_MODE_SHARED_LABEL,
        dtype: str = "auto",
        attn_implementation: str = "auto",
        x_vector_only_mode: bool = False,
        use_torch_compile: bool = False,
        use_cuda_graphs: bool = False,
        compile_mode: str = "reduce-overhead",
        asr_use_forced_aligner: bool = True,
        asr_translate_target_language: str = "English",
        asr_translate_instruction_override: str = DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE,
    ) -> tuple:
        """
        Create Qwen3-TTS engine configuration.

        Returns:
            Tuple containing engine config dict
        """
        runtime_mode = normalize_runtime_mode(runtime_mode)
        if runtime_mode == RUNTIME_MODE_SHARED:
            runtime_profile = "vibevoice_transformers4_shared"
        elif runtime_mode == RUNTIME_MODE_DEDICATED:
            runtime_profile = "qwen3_tts_transformers4_dedicated"
        else:
            runtime_profile = None

        resolved_model, model_name, model_spec = self._resolve_model(model_variant, voice_preset)
        model_type = model_spec["model_type"]
        model_size = model_spec["model_size"]
        model_role = model_spec["role"]

        if model_type == "CustomVoice" and voice_preset == "None (Zero-shot / Custom)":
            raise ValueError(
                "The selected Qwen CustomVoice model requires a preset speaker. "
                "Choose Vivian, Serena, or another preset in the Qwen3-TTS Engine."
            )
        effective_voice_preset = (
            voice_preset if model_type == "CustomVoice" else "None (Zero-shot / Custom)"
        )

        engine_config = {
            "engine_type": "qwen3_tts",
            "model_variant": resolved_model,
            "model_name": model_name,
            "model_path": resolved_model,
            "model_type": model_type,
            "model_role": model_role,
            "model_size": model_size,
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "voice_preset": effective_voice_preset,
            "language": language,
            "instruct": instruct,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "runtime_mode": runtime_mode,
            "runtime_profile": runtime_profile,
            "x_vector_only_mode": x_vector_only_mode,
            "use_torch_compile": use_torch_compile,
            "use_cuda_graphs": use_cuda_graphs,
            "compile_mode": compile_mode,
            "asr_use_forced_aligner": asr_use_forced_aligner,
            "asr_translate_target_language": asr_translate_target_language,
            "asr_translate_instruction_override": str(asr_translate_instruction_override or DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE).strip(),
        }

        # Print configuration summary (matching other engines)
        print(f"⚙️ Qwen3-TTS: Configured on {device}")
        print(f"   Model: {resolved_model} | Role: {model_role} | Language: {language}")
        runtime_label = {
            RUNTIME_MODE_MAIN: "Main Environment",
            RUNTIME_MODE_SHARED: "Shared Runtime",
            RUNTIME_MODE_DEDICATED: "Dedicated Runtime",
        }.get(runtime_mode, runtime_mode)
        print(f"   Runtime: {runtime_label}")
        print(f"   Settings: voice_preset={effective_voice_preset}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
        print(f"   Advanced: repetition_penalty={repetition_penalty}, max_tokens={max_new_tokens}, x_vector_only={x_vector_only_mode}")
        custom_asr_translate_instruction = (
            engine_config["asr_translate_instruction_override"] != DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE
        )
        print(
            f"   ASR: forced_aligner={asr_use_forced_aligner}, "
            f"translate_target={asr_translate_target_language}, "
            f"custom_translate_instruction={custom_asr_translate_instruction}"
        )
        if use_torch_compile or use_cuda_graphs:
            print(f"   Optimizations: torch.compile={use_torch_compile}, cuda_graphs={use_cuda_graphs}, mode={compile_mode}")
        if instruct:
            print(f"   Instruction: {instruct[:50]}..." if len(instruct) > 50 else f"   Instruction: {instruct}")
        if runtime_mode != RUNTIME_MODE_MAIN:
            print("   ⚠️ Runtime isolation requested: first run may create a secondary Qwen3-TTS runtime.")

        # Return in the same structure as other engines (nested config)
        engine_data = {
            "engine_type": "qwen3_tts",
            "config": engine_config,
            "capabilities": [model_role, "asr"]
        }

        return (engine_data,)
