"""
Granite ASR Engine Configuration Node

Provides Granite speech ASR/AST configuration for the unified ASR pipeline.
"""

import os
import sys
import importlib.util

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

from engines.granite_asr.prompting import DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE


class GraniteASREngineNode(BaseTTSNode):
    """
    Granite ASR engine configuration node.
    """

    SUPPORTED_LANGUAGES = [
        "Auto",
        "English",
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Japanese",
    ]

    @classmethod
    def NAME(cls):
        return "⚙️ Granite ASR Engine"

    @classmethod
    def INPUT_TYPES(cls):
        translate_languages = [
            "English",
            "French",
            "German",
            "Spanish",
            "Portuguese",
            "Japanese",
        ]
        return {
            "required": {
                "model_name": (["granite-4.0-1b-speech"], {
                    "default": "granite-4.0-1b-speech",
                    "tooltip": "Granite speech model:\n• granite-4.0-1b-speech: IBM 1B speech model for transcription and speech translation\n\nThis engine is ASR-only. It plugs into ✏️ ASR Transcribe like Qwen ASR does, but it does not generate native timestamps."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run Granite on:\n• auto: Best available device\n• cuda: NVIDIA GPU\n• cpu: CPU-only processing\n\nRecommended: auto unless you are debugging or intentionally forcing CPU."
                }),
                "max_new_tokens": ("INT", {
                    "default": 200, "min": 32, "max": 2048, "step": 32,
                    "tooltip": "Maximum text tokens Granite may generate per ASR chunk.\n\nThis is a hard cap, not a target:\n• Lower values: Safer against runaway output, but can cut off longer chunks\n• 200: Matches IBM's reference example\n• Higher values: Needed for longer spoken chunks, but can increase loop/repetition risk if the model goes off the rails"
                }),
            },
            "optional": {
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                    "tooltip": "Model precision:\n• auto: Prefer bfloat16 on capable CUDA GPUs, otherwise fall back safely\n• bfloat16: Best stability on supported GPUs\n• float16: Lower VRAM, wider GPU compatibility\n• float32: CPU-safe, highest VRAM/RAM use\n\nIf you hit strange numerical issues, try float32 on CPU or bfloat16 on newer GPUs."
                }),
                "attn_implementation": (["auto", "flash_attention_2", "sdpa", "eager"], {
                    "default": "auto",
                    "tooltip": "Attention backend:\n• auto: Try the fastest sane option, then fall back to eager if Granite's BLIP2 Q-Former rejects it\n• flash_attention_2: Fastest if installed and supported\n• sdpa: PyTorch native attention\n• eager: Slowest, most compatible fallback\n\nIf Granite fails to load with SDPA/Flash, eager is the compatibility escape hatch."
                }),
                "asr_use_forced_aligner": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use Qwen's separate forced aligner after Granite transcription so ✏️ ASR Transcribe can build word timings and SRT.\n\n• True: Enable timestamps/SRT support for Granite\n• False: Granite returns text only\n\nImportant:\n• Granite does NOT have native timestamp output in this integration\n• Translation mode still stays text-only\n• If ASR language is Auto, alignment uses a truthful heuristic: Japanese script -> Japanese mode, otherwise Qwen's generic space-delimited tokenizer path"
                }),
                "asr_translate_target_language": (translate_languages, {
                    "default": "English",
                    "tooltip": "Experimental ASR translation target for ✏️ ASR Transcribe when task=translate.\n\nImportant:\n• This is Granite ASR-only and only applies when the unified ASR node is set to translate\n• Granite target selection here is prompt-driven, not a native target-language API\n• Some language pairs may work better than others, and some may just fall back to transcription\n• English is the safest default; treat other targets as experimental until you validate them on real audio"
                }),
                "asr_translate_instruction_override": ("STRING", {
                    "default": DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE,
                    "multiline": True,
                    "tooltip": "Granite-only experimental translation instruction template for task=translate.\n\nThis field shows the actual default instruction Granite uses here. Edit it if you want to experiment.\n\nAvailable placeholders:\n• {source_language}: Replaced with the unified ASR source language, or 'the spoken source language' when ASR language is Auto\n• {target_language}: Replaced with this engine node's ASR translation target\n\nImportant:\n• This is NOT a raw chat template field. TTS Audio Suite still wraps it in Granite's chat format.\n• If you omit <|audio|>, TTS Audio Suite adds it for you.\n• Granite translation here is prompt-driven and can be inconsistent by language pair.\n• Weak or malformed instructions can be ignored and fall back to plain transcription."
                }),
                "do_sample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Native Hugging Face generation parameter.\n• False: Deterministic decoding, matches IBM's reference Granite example\n• True: Enable sampling-based decoding\n\nFor ASR, sampling can increase variation but also instability."
                }),
                "num_beams": ("INT", {
                    "default": 1, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Native beam search width:\n• 1: Greedy decoding\n• 2-4: Beam search, slower but can change transcript choices\n• Higher values: More expensive, rarely worth it for ASR unless you are experimenting"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Native sampling temperature.\n\nOnly really matters when do_sample is enabled:\n• Lower: More conservative token choices\n• Higher: More random output\n• 1.0: Neutral default"
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Native top-k sampling limit.\n• 0: Disable top-k filtering\n• Lower values: More constrained sampling\n• Higher values: Broader token pool\n\nMainly relevant when do_sample is enabled."
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Native top-p / nucleus sampling.\n• Lower values: More conservative sampling\n• 1.0: No nucleus restriction\n\nMainly relevant when do_sample is enabled."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05,
                    "tooltip": "Native repetition penalty.\n• 1.0: Disabled\n• >1.0: Discourage repeated token loops\n• <1.0: Encourages repetition and is usually a bad idea\n\nUseful when Granite starts spiraling into repeated phrases."
                }),
                "length_penalty": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Native beam-search length penalty.\n• <1.0: Bias shorter outputs\n• 1.0: Neutral\n• >1.0: Bias longer outputs\n\nMostly relevant with beam search, not plain greedy decoding."
                }),
                "no_repeat_ngram_size": ("INT", {
                    "default": 0, "min": 0, "max": 12, "step": 1,
                    "tooltip": "Native no-repeat n-gram guard.\n• 0: Disabled\n• 2-4: Blocks short repeated phrase patterns\n• Higher values: More aggressive repetition blocking, can overconstrain output"
                }),
                "early_stopping": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Native beam-search early stopping flag.\n• False: Let beam search continue normally\n• True: Stop once beams are considered complete\n\nOnly meaningful with num_beams > 1."
                }),
            }
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_name: str,
        device: str,
        max_new_tokens: int,
        dtype: str = "auto",
        attn_implementation: str = "auto",
        asr_use_forced_aligner: bool = True,
        asr_translate_target_language: str = "English",
        asr_translate_instruction_override: str = DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        early_stopping: bool = False,
    ):
        engine_config = {
            "engine_type": "granite_asr",
            "model_name": model_name,
            "device": device,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "max_new_tokens": int(max_new_tokens),
            "asr_use_forced_aligner": bool(asr_use_forced_aligner),
            "asr_translate_target_language": asr_translate_target_language,
            "asr_translate_instruction_override": str(asr_translate_instruction_override or DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE).strip(),
            "do_sample": bool(do_sample),
            "num_beams": int(num_beams),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "length_penalty": float(length_penalty),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
            "early_stopping": bool(early_stopping),
        }

        print(f"⚙️ Granite ASR: Configured on {device}")
        print(f"   Model: {model_name} | Max tokens: {max_new_tokens}")
        custom_translate_instruction = (
            engine_config["asr_translate_instruction_override"].strip() != DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE
        )
        print(
            f"   Advanced: dtype={dtype}, attn={attn_implementation}, "
            f"qwen_aligner={asr_use_forced_aligner}, translate_target={asr_translate_target_language}, "
            f"custom_translate_instruction={custom_translate_instruction}"
        )
        print(
            "   Decode: "
            f"do_sample={do_sample}, num_beams={num_beams}, temperature={temperature}, "
            f"top_k={top_k}, top_p={top_p}, repetition_penalty={repetition_penalty}, "
            f"length_penalty={length_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}, "
            f"early_stopping={early_stopping}"
        )

        return ({
            "engine_type": "granite_asr",
            "config": engine_config,
            "capabilities": ["asr"],
        },)
