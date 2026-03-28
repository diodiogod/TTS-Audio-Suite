"""
Unified ASR Transcribe Node - Engine-agnostic transcription for TTS Audio Suite.
"""

import os
import sys
import importlib.util
from typing import Any, Dict

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

BaseChatterBoxNode = base_module.BaseChatterBoxNode

from utils.asr.types import ASRRequest, asr_result_to_json
from utils.asr.pipeline import run_asr, format_asr_info, append_info_items
from utils.audio.audio_hash import generate_stable_audio_component


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

# Global cache for ASR results (audio hash + engine config + ASR params)
GLOBAL_ASR_CACHE = {}

class UnifiedASRTranscribeNode(BaseChatterBoxNode):
    @classmethod
    def NAME(cls):
        return "✏️ ASR Transcribe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": ("TTS_ENGINE", {
                    "tooltip": "ASR-capable engine configuration (for example Qwen3-TTS Engine or Granite ASR Engine). This node auto-routes to the correct ASR adapter based on the engine type."
                }),
                "audio": (any_typ, {
                    "tooltip": "Audio to transcribe. Accepts AUDIO, Character Voices output, or VideoHelper audio."
                }),
            },
            "optional": {
                "language": (["Auto"] + [
                    "Chinese",
                    "English",
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
                    "Macedonian"
                ], {
                    "default": "Auto",
                    "tooltip": "Language hint for ASR.\n• Auto: Let the engine handle language itself when possible\n• Explicit language: Better when you know the spoken language and want more predictable results\n\nEngine caveat:\n• Qwen ASR has native Auto language detection\n• Granite currently supports English, French, German, Spanish, Portuguese, and Japanese\n• Granite + forced aligner on Auto uses a truthful heuristic for timestamps: Japanese script -> Japanese mode, otherwise the generic space-delimited aligner path"
                }),
                "task": (["transcribe", "translate"], {
                    "default": "transcribe",
                    "tooltip": "Task mode:\n• transcribe: Same-language transcription\n• translate: Experimental speech translation to the engine's configured ASR translation target\n\nImportant:\n• Translation support varies a lot by engine and backend\n• In this repo, current ASR translation paths are prompt-driven rather than fully native task APIs\n• Expect uneven quality depending on language pair and model\n• Translation target is configured on the engine node, not here"
                }),
                "timestamps": (["none", "word"], {
                    "default": "none",
                    "tooltip": "Timing detail for the ASR timing output:\n• none: Text only, no reusable timed words/segments\n• word: Word-level timings for timestamp-capable ASR paths\n\nUse word timings if you plan to feed this into the Text to SRT Builder.\n\nGranite note: word timestamps are produced by the separate Qwen forced aligner, not natively by Granite."
                }),
                "chunk_size": ("INT", {
                    "default": 30, "min": 0, "max": 600, "step": 1,
                    "tooltip": "Chunk size in seconds. 0 = no chunking (use for short audio only)."
                }),
                "overlap": ("INT", {
                    "default": 2, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Overlap between chunks (seconds). Helps preserve words across chunk boundaries."
                }),
                "enable_asr_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache ASR results in memory so SRT tweaks are instant. Disable if you want fresh ASR every run."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "asr_timing_data", "info")
    FUNCTION = "transcribe"
    CATEGORY = "TTS Audio Suite/✏️ ASR"

    def transcribe(
        self,
        engine: Dict[str, Any],
        audio: Any,
        language: str = "Auto",
        task: str = "transcribe",
        timestamps: str = "none",
        chunk_size: int = 30,
        overlap: int = 2,
        enable_asr_cache: bool = True,
    ):
        if not isinstance(engine, dict):
            raise ValueError("Engine input must be a configuration dict")

        capabilities = engine.get("capabilities", [])
        if capabilities and "asr" not in capabilities:
            raise ValueError("Engine does not support ASR")

        audio_dict = self.normalize_audio_input(audio, "audio")

        # Warn if forced aligner is enabled but language is not supported by aligner
        if engine.get("config", engine).get("asr_use_forced_aligner", False):
            aligner_supported = {
                "Chinese", "English", "Cantonese", "French", "German",
                "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish"
            }
            if language != "Auto" and language not in aligner_supported:
                print(f"⚠️ Forced aligner may not support language '{language}'. "
                      f"Supported: {sorted(aligner_supported)}")
        forced_aligner_enabled = engine.get("config", engine).get("asr_use_forced_aligner", False)

        req = ASRRequest(
            audio=audio_dict,
            language=None if language == "Auto" else language,
            task=task,
            timestamps=timestamps,
            chunk_size=chunk_size,
            overlap=overlap,
            use_forced_aligner=forced_aligner_enabled,
        )

        cache_key = None
        if enable_asr_cache:
            audio_component = generate_stable_audio_component(reference_audio=audio_dict)
            engine_cfg = engine.get("config", engine)
            cache_data = {
                "engine_type": engine.get("engine_type"),
                "model_name": engine_cfg.get("model_name"),
                "model_size": engine_cfg.get("model_size"),
                "device": engine_cfg.get("device"),
                "dtype": engine_cfg.get("dtype"),
                "attn": engine_cfg.get("attn_implementation"),
                "max_new_tokens": engine_cfg.get("max_new_tokens"),
                "forced_aligner": engine_cfg.get("asr_use_forced_aligner", False),
                "asr_translate_target_language": engine_cfg.get("asr_translate_target_language"),
                "asr_translate_instruction_override": engine_cfg.get("asr_translate_instruction_override"),
                "do_sample": engine_cfg.get("do_sample"),
                "num_beams": engine_cfg.get("num_beams"),
                "temperature": engine_cfg.get("temperature"),
                "top_k": engine_cfg.get("top_k"),
                "top_p": engine_cfg.get("top_p"),
                "repetition_penalty": engine_cfg.get("repetition_penalty"),
                "length_penalty": engine_cfg.get("length_penalty"),
                "no_repeat_ngram_size": engine_cfg.get("no_repeat_ngram_size"),
                "early_stopping": engine_cfg.get("early_stopping"),
                "language": req.language,
                "task": req.task,
                "timestamps": req.timestamps,
                "chunk_size": req.chunk_size,
                "overlap": req.overlap,
                "audio_component": audio_component,
            }
            cache_key = str(sorted(cache_data.items()))
            cached = GLOBAL_ASR_CACHE.get(cache_key)
            if cached is not None:
                print("💾 CACHE HIT: Using cached ASR result")
                result = cached
            else:
                result = run_asr(engine, req)
                GLOBAL_ASR_CACHE[cache_key] = result
        else:
            result = run_asr(engine, req)

        info = format_asr_info(result)
        if timestamps == "word" and not result.segments:
            info = append_info_items(
                info,
                "WARNINGS",
                "Word timing data was requested, but this run returned no timed segments.",
            )
        elif timestamps == "none":
            info = append_info_items(
                info,
                "NOTES",
                "asr_timing_data has no word timings because timestamps=none.",
            )
        elif timestamps == "word" and not forced_aligner_enabled:
            info = append_info_items(
                info,
                "NOTES",
                "If you need high-quality subtitle timing from ASR output, enable the Qwen3 Forced Aligner when supported by the engine.",
            )

        return (result.text or "", asr_result_to_json(result), info)


NODE_CLASS_MAPPINGS = {
    "UnifiedASRTranscribeNode": UnifiedASRTranscribeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedASRTranscribeNode": "✏️ ASR Transcribe"
}
