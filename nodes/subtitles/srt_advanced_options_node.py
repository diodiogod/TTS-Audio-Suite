"""
SRT Advanced Options Node - Fine-tune subtitle construction behavior.
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

from utils.asr.srt_heuristic_profiles import (
    DEFAULT_HEURISTIC_PROFILE_LABEL,
    ENGLISH_DANGLING_TAIL_ALLOWLIST,
    ENGLISH_INCOMPLETE_KEYWORDS,
    HEURISTIC_PROFILE_OPTIONS,
)


class SRTAdvancedOptionsNode(BaseChatterBoxNode):
    @classmethod
    def NAME(cls):
        return "🔧 SRT Advanced Options"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_preset": (["Custom", "Netflix-Standard", "Broadcast", "Fast speech", "Mobile", "TTS-Ready", "TTS-Ready (Paragraphs)"], {
                    "default": "Broadcast",
                    "tooltip": "Readability preset for subtitle building.\nChoose a preset to seed the knobs below with recommended values, then edit them as needed.\nIf you change a preset-derived knob, the UI will switch to Custom automatically.\n\nExamples:\n• Broadcast: conservative timing, safe desktop readability\n• Netflix-Standard: similar readability with longer max duration\n• Fast speech: denser subtitles for rapid speech\n• Mobile: shorter lines for smaller screens\n• TTS-Ready: single-line cues that stop by meaning instead of display wrapping\n• TTS-Ready (Paragraphs): same TTS-ready behavior, but tuned for longer paragraph-sized cues"
                }),
                "srt_mode": (["smart", "engine_segments", "words"], {
                    "default": "smart",
                    "tooltip": "How subtitle cues are built:\n• smart: rebuild from word timings for readability\n• engine_segments: trust incoming segments as-is\n• words: one word per cue (debug/alignment only)\n\nUse smart unless you have a reason not to."
                }),
                "tts_ready_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Build cues for downstream TTS instead of on-screen subtitles.\nThis disables multi-line display wrapping pressure, keeps each cue on one line, and prefers semantic stopping points over character-count stops."
                }),
                "tts_ready_paragraph_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Only used when TTS-ready is enabled.\nPrefer one cue per paragraph and only split if a paragraph is genuinely too long for clean TTS playback."
                }),
                "heuristic_language_profile": (HEURISTIC_PROFILE_OPTIONS, {
                    "default": DEFAULT_HEURISTIC_PROFILE_LABEL,
                    "tooltip": "Language profile for heuristic defaults.\nPick a language to auto-populate connector and incomplete-sentence lists.\nAuto resolves from ASR timing language when available. Custom means you fully manage the text lists yourself."
                }),
                "srt_max_chars_per_line": ("INT", {
                    "default": 42, "min": 10, "max": 10000, "step": 1,
                    "tooltip": "Maximum characters per subtitle line.\nLower = shorter lines, more splits.\nTypical values: 32 mobile, 42 desktop/broadcast."
                }),
                "srt_max_lines": ("INT", {
                    "default": 2, "min": 1, "max": 3, "step": 1,
                    "tooltip": "Maximum lines per subtitle cue.\n2 is the normal default. 3 is denser but harder to read."
                }),
                "srt_max_duration": ("FLOAT", {
                    "default": 6.0, "min": 0.2, "max": 9999.0, "step": 0.1,
                    "tooltip": "Maximum on-screen duration for a subtitle cue in seconds.\nHigher = fewer splits; too high feels laggy."
                }),
                "srt_min_duration": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.1,
                    "tooltip": "Minimum on-screen duration in seconds.\nHigher = fewer flash cues; lower = tighter sync."
                }),
                "srt_min_gap": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 9999.0, "step": 0.1,
                    "tooltip": "Pause length that forces a new subtitle cue.\nHigher = more merging across short pauses."
                }),
                "srt_max_cps": ("FLOAT", {
                    "default": 20.0, "min": 0.1, "max": 9999.0, "step": 0.5,
                    "tooltip": "Maximum reading speed in characters per second.\nLower = easier reading, more splits.\nHigher = denser subtitles."
                }),
                "dedupe_overlaps": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove overlapping duplicate phrases from bad word timing data.\nUseful for alignment glitches.\nCan also remove real repetitions like choruses."
                }),
                "dedupe_window_ms": ("INT", {
                    "default": 1500, "min": 0, "max": 10000, "step": 50,
                    "tooltip": "Time window used to detect overlapping duplicates in milliseconds.\nHigher = more aggressive dedupe."
                }),
                "dedupe_min_words": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Minimum matching word count before a repeated phrase is considered a duplicate.\nHigher = safer."
                }),
                "dedupe_overlap_ratio": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Required timing overlap ratio before duplicate text is removed.\nHigher = stricter dedupe."
                }),
                "punctuation_grace_chars": ("INT", {
                    "default": 12, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Allow a sentence-ending punctuation mark to exceed the max line length by this many chars.\nHelps avoid ugly breaks right before punctuation."
                }),
                "min_words_per_segment": ("INT", {
                    "default": 2, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Merge very tiny subtitle segments into neighbors.\nHigher = fewer one-word cues."
                }),
                "min_segment_seconds": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Merge subtitle cues shorter than this duration.\nHigher = fewer micro-cues."
                }),
                "merge_trailing_punct_word": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep a trailing word with punctuation attached to the previous subtitle when possible.\nFixes splits like \"beautiful / world.\""
                }),
                "merge_trailing_punct_max_gap": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Maximum pause allowed when bridging that trailing punctuation word.\nHigher = more aggressive bridging."
                }),
                "merge_leading_short_phrase": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge a very short phrase into the previous cue when it follows punctuation.\nFixes splits like \"I'm a / riddle.\""
                }),
                "merge_leading_short_max_words": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Maximum word count for that short leading phrase.\nHigher = more aggressive merging."
                }),
                "merge_leading_short_max_gap": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Maximum pause allowed when merging a short leading phrase.\nHigher = more merging across pauses."
                }),
                "merge_dangling_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge a short dangling ending into the next subtitle when it ends on a connector word.\nUseful for incomplete fragments."
                }),
                "merge_dangling_tail_max_words": ("INT", {
                    "default": 3, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Maximum words allowed in that dangling ending.\nHigher = more aggressive merging."
                }),
                "merge_dangling_tail_max_gap": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 6.0, "step": 0.05,
                    "tooltip": "Maximum pause allowed when merging a dangling tail.\nHigher = more aggressive merging."
                }),
                "merge_dangling_tail_allowlist": ("STRING", {
                    "default": ENGLISH_DANGLING_TAIL_ALLOWLIST,
                    "tooltip": "Comma-separated connector words treated as dangling tails.\nExample: a, the, to, of, and, I'm"
                }),
                "merge_leading_short_no_punct": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge a very short follow-up into the previous subtitle even without punctuation.\nUseful for awkward mid-thought splits."
                }),
                "merge_leading_short_no_punct_max_words": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Maximum words in that short follow-up.\nHigher = more aggressive merging."
                }),
                "merge_leading_short_no_punct_max_gap": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Maximum pause allowed when merging that follow-up.\nHigher = more aggressive merging."
                }),
                "merge_incomplete_sentence": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge short continuations when the previous subtitle clearly looks incomplete.\nUseful for broken questions and sentence fragments."
                }),
                "merge_incomplete_max_gap": ("FLOAT", {
                    "default": 1.2, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Maximum pause allowed when merging an incomplete sentence.\nHigher = more aggressive merging."
                }),
                "merge_incomplete_keywords": ("STRING", {
                    "default": ENGLISH_INCOMPLETE_KEYWORDS,
                    "tooltip": "Comma-separated keywords that suggest the previous subtitle is incomplete.\nExample: what, why, how, where"
                }),
                "merge_incomplete_split_next": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If the next subtitle contains multiple sentences, split it and only merge the first sentence.\nHelps keep merged subtitles readable."
                }),
                "merge_allow_overlong": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow merges even if the final subtitle exceeds max duration.\nGood for songs and slow speech. Disable for strict timing limits."
                }),
                "normalize_cue_end_punctuation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Optional subtitle-style cleanup.\nWhen enabled, removes trailing commas, periods, semicolons, and colons at the visual end of a subtitle cue.\nIf a cue is cleaned this way, the next cue start is uppercased to keep the subtitle flow visually coherent.\n\nQuestion marks, exclamation points, and ellipses are preserved.\nThis is a style transform, not grammatical truth, so it stays disabled by default."
                }),
            }
        }

    RETURN_TYPES = ("SRT_OPTIONS",)
    RETURN_NAMES = ("srt_options",)
    FUNCTION = "build_options"
    CATEGORY = "TTS Audio Suite/📺 Subtitles"

    def build_options(
        self,
        srt_preset: str,
        srt_mode: str,
        tts_ready_mode: bool,
        tts_ready_paragraph_mode: bool,
        heuristic_language_profile: str,
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
        normalize_cue_end_punctuation: bool,
    ):
        return ({
            "srt_preset": srt_preset,
            "srt_mode": srt_mode,
            "tts_ready_mode": tts_ready_mode,
            "heuristic_language_profile": heuristic_language_profile,
            "srt_max_chars_per_line": srt_max_chars_per_line,
            "srt_max_lines": srt_max_lines,
            "srt_max_duration": srt_max_duration,
            "srt_min_duration": srt_min_duration,
            "srt_min_gap": srt_min_gap,
            "srt_max_cps": srt_max_cps,
            "tts_ready_paragraph_mode": tts_ready_paragraph_mode,
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
            "normalize_cue_end_punctuation": normalize_cue_end_punctuation,
        },)

NODE_CLASS_MAPPINGS = {
    "SRTAdvancedOptionsNode": SRTAdvancedOptionsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SRTAdvancedOptionsNode": "🔧 SRT Advanced Options",
}
