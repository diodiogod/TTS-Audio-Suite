"""
Qwen3-TTS Engine Configuration Node

Provides unified configuration interface for Qwen3-TTS engine with intelligent
model selection based on voice preset choice (CustomVoice/VoiceDesign/Base).
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
from utils.models.extra_paths import get_all_tts_model_paths


class Qwen3TTSEngineNode(BaseTTSNode):
    """
    Qwen3-TTS Engine configuration node.
    Unified interface for all 3 model variants (CustomVoice/VoiceDesign/Base).
    """

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

        return {
            "required": {
                # Model Configuration
                "model_size": (["1.7B", "0.6B"], {
                    "default": "1.7B",
                    "tooltip": "Model size:\n• 1.7B: High quality, supports all features (~12GB VRAM)\n• 0.6B: Low VRAM, no instruction support (~6GB VRAM)\nNote: VoiceDesign requires 1.7B (auto-switches)"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run Qwen3-TTS model on:\n• auto: Automatically select best available\n• cuda: NVIDIA GPU (requires CUDA)\n• cpu: CPU-only processing (very slow)"
                }),

                # Voice Control
                "voice_preset": (voice_presets, {
                    "default": "None (Zero-shot / Custom)",
                    "tooltip": "Voice selection:\n• None: Zero-shot voice cloning from reference audio (Base model)\n• Preset names: Use hardcoded voices (CustomVoice model)\nModel is auto-selected based on this choice"
                }),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {
                    "default": "Auto",
                    "tooltip": "Target language for speech generation:\n• Auto: Automatically detect from text\n• Portuguese: European Portuguese only for Base model\n  - Brazilian Portuguese: Only available with CustomVoice presets + instruction\n  - Example instruction: 'Speak with Brazilian Portuguese accent'"
                }),
                "instruct": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Instruction for emotion/style/accent control (CustomVoice with presets only):\n• Emotion: 'Speak slowly with sadness', 'Energetic and excited'\n• Accent: 'Speak with Brazilian Portuguese accent'\n• Works best in English instructions\n• LOCKED when:\n  - voice_preset = None (Base model has no instruction)\n  - model_size = 0.6B + preset (0.6B CustomVoice has no instruction)\n• UNLOCKED when: 1.7B + preset selected"
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
                    "tooltip": "Enable torch.compile for decoder (~1.5-2x speedup):\n• False: Standard inference (RECOMMENDED - works with all PyTorch versions)\n• True: Compiled decoder (REQUIRES PyTorch 2.10+ and triton-windows 3.6+)\n⚠️ REQUIREMENTS: PyTorch 2.10.0+cu130, triton-windows 3.6.0+\n⚠️ See docs/qwen3_tts_optimizations.md for installation\nFirst generation slower due to compilation, then ~1.5-2x faster"
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
                    "default": False,
                    "tooltip": "Enable Qwen3 forced aligner (required for word timestamps + accurate SRT).\nIf OFF: ASR text works, but SRT output is limited and will show a warning."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_size: str,
        device: str,
        voice_preset: str,
        language: str,
        instruct: str,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        dtype: str = "auto",
        attn_implementation: str = "auto",
        x_vector_only_mode: bool = False,
        use_torch_compile: bool = False,
        use_cuda_graphs: bool = False,
        compile_mode: str = "reduce-overhead",
        asr_use_forced_aligner: bool = False,
    ) -> tuple:
        """
        Create Qwen3-TTS engine configuration.

        Returns:
            Tuple containing engine config dict
        """
        # Create engine config dict
        # NOTE: model_path is NOT included - adapter determines correct model variant automatically
        # based on voice_preset and model_size (CustomVoice/VoiceDesign/Base)
        engine_config = {
            "engine_type": "qwen3_tts",
            "model_size": model_size,
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "voice_preset": voice_preset,
            "language": language,
            "instruct": instruct,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "x_vector_only_mode": x_vector_only_mode,
            "use_torch_compile": use_torch_compile,
            "use_cuda_graphs": use_cuda_graphs,
            "compile_mode": compile_mode,
            "asr_use_forced_aligner": asr_use_forced_aligner,
        }

        # Print configuration summary (matching other engines)
        print(f"⚙️ Qwen3-TTS: Configured on {device}")
        print(f"   Model: {model_size} | Language: {language}")
        print(f"   Settings: voice_preset={voice_preset}, temperature={temperature}, top_k={top_k}, top_p={top_p}")
        print(f"   Advanced: repetition_penalty={repetition_penalty}, max_tokens={max_new_tokens}, x_vector_only={x_vector_only_mode}")
        if use_torch_compile or use_cuda_graphs:
            print(f"   Optimizations: torch.compile={use_torch_compile}, cuda_graphs={use_cuda_graphs}, mode={compile_mode}")
        if instruct:
            print(f"   Instruction: {instruct[:50]}..." if len(instruct) > 50 else f"   Instruction: {instruct}")

        # Return in the same structure as other engines (nested config)
        engine_data = {
            "engine_type": "qwen3_tts",
            "config": engine_config,
            "capabilities": ["tts", "asr"]
        }

        return (engine_data,)
