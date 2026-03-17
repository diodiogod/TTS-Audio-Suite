"""
Standalone punctuation / truecase cleanup for raw ASR text.
"""

from utils.aux_models.registry import get_punctuation_dropdown_labels
from utils.text.punctuation_runtime import restore_punctuation


class ASRPunctuationTruecaseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "this was designed mainly for raw asr outputs like granite but it can clean up any plain text transcript as well",
                    "tooltip": "Raw text to post-process with punctuation and true-casing.\n\nDesigned mainly for low-punctuation ASR outputs such as Granite transcripts, but it can be used on any plain text transcript.\n\nImportant: this expects plain transcript text, not full SRT files with cue numbers and timestamps.\n\nBest results usually come from lowercased, lightly cleaned ASR text rather than already-punctuated prose."
                }),
                "model": (get_punctuation_dropdown_labels(), {
                    "default": get_punctuation_dropdown_labels()[0],
                    "tooltip": "Choose the punctuation/truecase model.\n\nEnglish is the lightest option.\nRomance covers ES/FR/IT/PT/CA/GL.\nXLM-R is the heavy multilingual fallback.\n\nThese are helper models, not ASR engines."
                }),
                "processing_scope": (["Whole Text", "Per Paragraph"], {
                    "default": "Whole Text",
                    "tooltip": "How to feed text to the punctuation model.\n\nWhole Text: process the entire input as one block.\nPer Paragraph: split on blank lines first and preserve paragraph breaks."
                }),
                "output_mode": (["Restored Paragraphs", "One Sentence Per Line"], {
                    "default": "Restored Paragraphs",
                    "tooltip": "How to format the output text.\n\nRestored Paragraphs: normal punctuated text.\nOne Sentence Per Line: useful for subtitle prep, manual review, or later segmentation."
                }),
                "lowercase_input_first": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Lowercase First",
                    "label_off": "Keep Original Casing",
                    "tooltip": "Lowercase the input before running the punctuation model.\n\nRecommended for raw ASR text because these helper models are mainly trained for lowercased, unpunctuated transcripts.\nDisable only if you are intentionally experimenting with already-cased input."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "info")
    FUNCTION = "restore"
    CATEGORY = "TTS Audio Suite/Text"

    def restore(
        self,
        text: str,
        model: str,
        processing_scope: str,
        output_mode: str,
        lowercase_input_first: bool,
    ):
        restored_text, info = restore_punctuation(
            text=text,
            model_label=model,
            processing_scope=processing_scope,
            output_mode=output_mode,
            lowercase_input_first=lowercase_input_first,
        )

        print(f"📝 Punctuation / Truecase: {model}")
        print(f"   Scope: {processing_scope} | Output: {output_mode} | Lowercase first: {lowercase_input_first}")

        return restored_text, info
