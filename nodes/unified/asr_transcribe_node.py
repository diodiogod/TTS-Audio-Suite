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

from utils.asr.types import ASRRequest
from utils.asr.pipeline import run_asr, format_asr_output
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
                "engine": (any_typ, {
                    "tooltip": "ASR-capable engine configuration (use Qwen3 Engine output). This node auto-routes to the correct ASR adapter based on the engine type."
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
                    "tooltip": "Language hint for ASR. Auto lets the model detect language. This list matches Qwen3-ASR supported languages."
                }),
                "task": (["transcribe", "translate"], {
                    "default": "transcribe",
                    "tooltip": "Task mode:\n• transcribe: Same-language transcription\n• translate: Translate speech to English (engine support varies)"
                }),
                "timestamps": (["none", "word"], {
                    "default": "none",
                    "tooltip": "Timestamp output:\n• none: Text only\n• word: Word-level timestamps (requires forced aligner support)"
                }),
                "srt_options": ("ASR_SRT_OPTIONS", {
                    "tooltip": "Advanced SRT construction options (optional)."
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "srt", "timestamps", "info")
    FUNCTION = "transcribe"
    CATEGORY = "TTS Audio Suite/✏️ ASR"

    def transcribe(
        self,
        engine: Dict[str, Any],
        audio: Any,
        language: str = "Auto",
        task: str = "transcribe",
        timestamps: str = "none",
        srt_options: Dict[str, Any] = None,
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

        req = ASRRequest(
            audio=audio_dict,
            language=None if language == "Auto" else language,
            task=task,
            timestamps=timestamps,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        cache_key = None
        if enable_asr_cache:
            audio_component = generate_stable_audio_component(reference_audio=audio_dict)
            engine_cfg = engine.get("config", engine)
            cache_data = {
                "engine_type": engine.get("engine_type"),
                "model_size": engine_cfg.get("model_size"),
                "device": engine_cfg.get("device"),
                "dtype": engine_cfg.get("dtype"),
                "attn": engine_cfg.get("attn_implementation"),
                "max_new_tokens": engine_cfg.get("max_new_tokens"),
                "forced_aligner": engine_cfg.get("asr_use_forced_aligner", False),
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

        # Default SRT options (Broadcast preset)
        opts = {
            "srt_preset": "Broadcast",
            "srt_mode": "smart",
            "srt_max_chars_per_line": 42,
            "srt_max_lines": 2,
            "srt_max_duration": 6.0,
            "srt_min_duration": 1.0,
            "srt_min_gap": 0.6,
            "srt_max_cps": 17.0,
            "dedupe_overlaps": True,
            "dedupe_window_ms": 1500,
            "dedupe_min_words": 2,
            "dedupe_overlap_ratio": 0.6,
            "punctuation_grace_chars": 12,
            "min_words_per_segment": 2,
            "min_segment_seconds": 0.4,
            "merge_trailing_punct_word": True,
            "merge_trailing_punct_max_gap": 1.0,
            "merge_leading_short_phrase": True,
            "merge_leading_short_max_words": 2,
            "merge_leading_short_max_gap": 2.0,
            "merge_dangling_tail": True,
            "merge_dangling_tail_max_words": 3,
            "merge_dangling_tail_max_gap": 3.0,
            "merge_dangling_tail_allowlist": "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
            "merge_leading_short_no_punct": True,
            "merge_leading_short_no_punct_max_words": 2,
            "merge_leading_short_no_punct_max_gap": 1.5,
            "merge_incomplete_sentence": True,
            "merge_incomplete_max_gap": 1.2,
            "merge_incomplete_keywords": "what,why,how,where,who,which,when",
            "merge_incomplete_split_next": True,
            "merge_allow_overlong": True,
        }

        if isinstance(srt_options, dict):
            opts.update(srt_options)

        # Apply preset overrides
        preset = opts.get("srt_preset", "Broadcast")
        if preset != "Custom":
            if preset == "Netflix-Standard":
                opts.update({
                    "srt_max_chars_per_line": 42,
                    "srt_max_lines": 2,
                    "srt_max_duration": 7.0,
                    "srt_min_duration": 0.85,
                    "srt_min_gap": 0.2,
                    "srt_max_cps": 17.0,
                })
            elif preset == "Fast speech":
                opts.update({
                    "srt_max_chars_per_line": 42,
                    "srt_max_lines": 2,
                    "srt_max_duration": 6.0,
                    "srt_min_duration": 0.8,
                    "srt_min_gap": 0.4,
                    "srt_max_cps": 20.0,
                })
            elif preset == "Mobile":
                opts.update({
                    "srt_max_chars_per_line": 32,
                    "srt_max_lines": 2,
                    "srt_max_duration": 5.0,
                    "srt_min_duration": 1.0,
                    "srt_min_gap": 0.6,
                    "srt_max_cps": 17.0,
                })
            else:
                opts.update({
                    "srt_max_chars_per_line": 42,
                    "srt_max_lines": 2,
                    "srt_max_duration": 6.0,
                    "srt_min_duration": 1.0,
                    "srt_min_gap": 0.6,
                    "srt_max_cps": 17.0,
                })

        out = format_asr_output(
            result,
            srt_mode=opts["srt_mode"],
            max_chars_per_line=opts["srt_max_chars_per_line"],
            max_lines=opts["srt_max_lines"],
            max_duration=opts["srt_max_duration"],
            min_duration=opts["srt_min_duration"],
            min_gap=opts["srt_min_gap"],
            max_cps=opts["srt_max_cps"],
            dedupe_overlaps=opts["dedupe_overlaps"],
            dedupe_window_ms=opts["dedupe_window_ms"],
            dedupe_min_words=opts["dedupe_min_words"],
            dedupe_overlap_ratio=opts["dedupe_overlap_ratio"],
            punctuation_grace_chars=opts["punctuation_grace_chars"],
            min_words_per_segment=opts["min_words_per_segment"],
            min_segment_seconds=opts["min_segment_seconds"],
            merge_trailing_punct_word=opts["merge_trailing_punct_word"],
            merge_trailing_punct_max_gap=opts["merge_trailing_punct_max_gap"],
            merge_leading_short_phrase=opts["merge_leading_short_phrase"],
            merge_leading_short_max_words=opts["merge_leading_short_max_words"],
            merge_leading_short_max_gap=opts["merge_leading_short_max_gap"],
            merge_dangling_tail=opts["merge_dangling_tail"],
            merge_dangling_tail_max_words=opts["merge_dangling_tail_max_words"],
            merge_dangling_tail_max_gap=opts["merge_dangling_tail_max_gap"],
            merge_dangling_tail_allowlist=opts["merge_dangling_tail_allowlist"],
            merge_leading_short_no_punct=opts["merge_leading_short_no_punct"],
            merge_leading_short_no_punct_max_words=opts["merge_leading_short_no_punct_max_words"],
            merge_leading_short_no_punct_max_gap=opts["merge_leading_short_no_punct_max_gap"],
            merge_incomplete_sentence=opts["merge_incomplete_sentence"],
            merge_incomplete_max_gap=opts["merge_incomplete_max_gap"],
            merge_incomplete_keywords=opts["merge_incomplete_keywords"],
            merge_incomplete_split_next=opts["merge_incomplete_split_next"],
            merge_allow_overlong=opts["merge_allow_overlong"],
        )
        return (out["text"], out["srt"], out["timestamps"], out["info"])


NODE_CLASS_MAPPINGS = {
    "UnifiedASRTranscribeNode": UnifiedASRTranscribeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedASRTranscribeNode": "✏️ ASR Transcribe"
}
