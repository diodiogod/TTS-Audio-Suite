"""
ASR SRT Advanced Options Node - Fine-tune subtitle construction for ASR outputs.
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

BaseChatterBoxNode = base_module.BaseChatterBoxNode


class ASRSRTAdvancedOptionsNode(BaseChatterBoxNode):
    @classmethod
    def NAME(cls):
        return "🔧 ASR SRT Advanced Options"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_preset": (["Custom", "Netflix-Standard", "Broadcast", "Fast speech", "Mobile"], {
                    "default": "Broadcast",
                    "tooltip": "Readability preset (standards-style).\nCustom = you control everything below.\n\n📌 Examples (approx):\n• Broadcast: 42 CPL, 17 CPS, 6s max\n• Netflix-Standard: 42 CPL, 17 CPS, 7s max\n• Fast speech: 42 CPL, 20 CPS, 6s max\n• Mobile: 32 CPL, 17 CPS, 5s max\n\nTip: Presets aim for safe industry readability; advanced controls let you break the rules."
                }),
                "srt_mode": (["smart", "engine_segments", "words"], {
                    "default": "smart",
                    "tooltip": "How subtitles are built:\n• smart: re-segment words for readability (recommended)\n• engine_segments: trust model segments as-is\n• words: one word per subtitle (debug / alignment)"
                }),
                "srt_max_chars_per_line": ("INT", {
                    "default": 42, "min": 10, "max": 10000, "step": 1,
                    "tooltip": "Max characters per line.\nHigher = longer lines, fewer splits.\nExample: 32–36 mobile, 42 desktop."
                }),
                "srt_max_lines": ("INT", {
                    "default": 2, "min": 1, "max": 3, "step": 1,
                    "tooltip": "Max lines per subtitle block.\nHigher = taller subtitles, fewer splits.\nExample: 2 standard, 3 dense speech."
                }),
                "srt_max_duration": ("FLOAT", {
                    "default": 6.0, "min": 0.2, "max": 9999.0, "step": 0.1,
                    "tooltip": "Max time a subtitle stays on screen (seconds).\nHigher = fewer splits; too high can feel laggy."
                }),
                "srt_min_duration": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.1,
                    "tooltip": "Minimum time on screen (seconds).\nHigher = fewer short subtitles; lower = more rapid changes."
                }),
                "srt_min_gap": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 9999.0, "step": 0.1,
                    "tooltip": "Pause length that forces a new subtitle (seconds).\nHigher = fewer splits on short pauses."
                }),
                "srt_max_cps": ("FLOAT", {
                    "default": 20.0, "min": 0.1, "max": 9999.0, "step": 0.5,
                    "tooltip": "Reading speed limit (characters per second).\nLower = easier to read, more splits.\nHigher = denser, harder to read."
                }),
                "dedupe_overlaps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove overlapping duplicate phrases from bad alignments.\nMay also remove real repeats (chorus)."
                }),
                "dedupe_window_ms": ("INT", {
                    "default": 1500, "min": 0, "max": 10000, "step": 50,
                    "tooltip": "Time window to detect overlaps (ms).\nHigher = more aggressive duplicate removal."
                }),
                "dedupe_min_words": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Minimum matching words to consider a repeat.\nHigher = safer; lower = more aggressive."
                }),
                "dedupe_overlap_ratio": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "How much timing overlap is required to drop a duplicate phrase.\nHigher = stricter."
                }),
                "punctuation_grace_chars": ("INT", {
                    "default": 12, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Let a sentence end (.,!,?,…) exceed max length by this many chars.\nHigher = fewer awkward breaks before punctuation."
                }),
                "min_words_per_segment": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Merge very short segments into the next one.\nHigher = fewer tiny 1–2 word subtitles."
                }),
                "min_segment_seconds": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Merge segments shorter than this (seconds).\nHigher = fewer micro subtitles; too high can blur timing."
                }),
                "merge_trailing_punct_word": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep a trailing word with punctuation in the previous subtitle.\nFixes cases like \"beautiful / world.\" across a short pause."
                }),
                "merge_trailing_punct_max_gap": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Max pause allowed when bridging a trailing punctuation word (seconds).\nHigher = more bridging across pauses."
                }),
                "merge_leading_short_phrase": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge a very short phrase into the previous subtitle if it follows punctuation.\nFixes cases like \"I'm a / riddle.\""
                }),
                "merge_leading_short_max_words": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Max words to treat as a short leading phrase.\nLower = safer, higher = more aggressive."
                }),
                "merge_leading_short_max_gap": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Max pause allowed when merging a short leading phrase (seconds).\nHigher = more merging across pauses."
                }),
                "merge_dangling_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge a short “hanging” ending into the next subtitle when it ends on a connector word.\nExample: \"I'm a / riddle.\""
                }),
                "merge_dangling_tail_max_words": ("INT", {
                    "default": 3, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Max words allowed in that hanging ending.\nHigher = more aggressive merges."
                }),
                "merge_dangling_tail_max_gap": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 6.0, "step": 0.05,
                    "tooltip": "Max pause allowed when merging a dangling tail (seconds).\nHigher = more merging across pauses."
                }),
                "merge_dangling_tail_allowlist": ("STRING", {
                    "default": "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
                    "tooltip": "Comma list of connector words that count as dangling tails.\nExample: a, the, to, of, and, I'm"
                }),
                "merge_leading_short_no_punct": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge a very short follow-up into the previous subtitle even without punctuation.\nFixes cases like \"What the hell / am I\"."
                }),
                "merge_leading_short_no_punct_max_words": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Max words in that short follow-up.\nHigher = more aggressive merges."
                }),
                "merge_leading_short_no_punct_max_gap": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Max pause allowed when merging that follow-up (seconds).\nHigher = more merging across pauses."
                }),
                "merge_incomplete_sentence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge short continuations of a question when the previous line looks incomplete.\nExample: \"What the hell / am I doing here?\""
                }),
                "merge_incomplete_max_gap": ("FLOAT", {
                    "default": 1.2, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Max pause allowed when merging an incomplete sentence (seconds).\nHigher = more merging across pauses."
                }),
                "merge_incomplete_keywords": ("STRING", {
                    "default": "what,why,how,where,who,which,when",
                    "tooltip": "Comma list of question keywords that suggest an incomplete sentence.\nExample: what, why, how, where"
                }),
                "merge_incomplete_split_next": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If the next subtitle has multiple sentences, split it and only merge the first sentence.\nHelps keep lines short and readable."
                }),
                "merge_allow_overlong": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow merges even if they exceed max duration.\nBest for songs or slow speech; disable for strict timing."
                }),
            }
        }

    RETURN_TYPES = ("ASR_SRT_OPTIONS",)
    RETURN_NAMES = ("asr_srt_options",)
    FUNCTION = "build_options"
    CATEGORY = "TTS Audio Suite/✏️ ASR"

    def build_options(
        self,
        srt_preset: str,
        srt_mode: str,
        srt_max_chars_per_line: int,
        srt_max_lines: int,
        srt_max_duration: float,
        srt_min_duration: float,
        srt_min_gap: float,
        srt_max_cps: float,
        dedupe_overlaps: bool,
        dedupe_window_ms: int,
        dedupe_min_words: int,
        dedupe_overlap_ratio: float,
        punctuation_grace_chars: int,
        min_words_per_segment: int,
        min_segment_seconds: float,
        merge_trailing_punct_word: bool,
        merge_trailing_punct_max_gap: float,
        merge_leading_short_phrase: bool,
        merge_leading_short_max_words: int,
        merge_leading_short_max_gap: float,
        merge_dangling_tail: bool,
        merge_dangling_tail_max_words: int,
        merge_dangling_tail_max_gap: float,
        merge_dangling_tail_allowlist: str,
        merge_leading_short_no_punct: bool,
        merge_leading_short_no_punct_max_words: int,
        merge_leading_short_no_punct_max_gap: float,
        merge_incomplete_sentence: bool,
        merge_incomplete_max_gap: float,
        merge_incomplete_keywords: str,
        merge_incomplete_split_next: bool,
        merge_allow_overlong: bool,
    ):
        return ({
            "srt_preset": srt_preset,
            "srt_mode": srt_mode,
            "srt_max_chars_per_line": srt_max_chars_per_line,
            "srt_max_lines": srt_max_lines,
            "srt_max_duration": srt_max_duration,
            "srt_min_duration": srt_min_duration,
            "srt_min_gap": srt_min_gap,
            "srt_max_cps": srt_max_cps,
            "dedupe_overlaps": dedupe_overlaps,
            "dedupe_window_ms": dedupe_window_ms,
            "dedupe_min_words": dedupe_min_words,
            "dedupe_overlap_ratio": dedupe_overlap_ratio,
            "punctuation_grace_chars": punctuation_grace_chars,
            "min_words_per_segment": min_words_per_segment,
            "min_segment_seconds": min_segment_seconds,
            "merge_trailing_punct_word": merge_trailing_punct_word,
            "merge_trailing_punct_max_gap": merge_trailing_punct_max_gap,
            "merge_leading_short_phrase": merge_leading_short_phrase,
            "merge_leading_short_max_words": merge_leading_short_max_words,
            "merge_leading_short_max_gap": merge_leading_short_max_gap,
            "merge_dangling_tail": merge_dangling_tail,
            "merge_dangling_tail_max_words": merge_dangling_tail_max_words,
            "merge_dangling_tail_max_gap": merge_dangling_tail_max_gap,
            "merge_dangling_tail_allowlist": merge_dangling_tail_allowlist,
            "merge_leading_short_no_punct": merge_leading_short_no_punct,
            "merge_leading_short_no_punct_max_words": merge_leading_short_no_punct_max_words,
            "merge_leading_short_no_punct_max_gap": merge_leading_short_no_punct_max_gap,
            "merge_incomplete_sentence": merge_incomplete_sentence,
            "merge_incomplete_max_gap": merge_incomplete_max_gap,
            "merge_incomplete_keywords": merge_incomplete_keywords,
            "merge_incomplete_split_next": merge_incomplete_split_next,
            "merge_allow_overlong": merge_allow_overlong,
        },)


NODE_CLASS_MAPPINGS = {
    "ASRSRTAdvancedOptionsNode": ASRSRTAdvancedOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASRSRTAdvancedOptionsNode": "🔧 ASR SRT Advanced Options"
}
