"""
Shared Granite ASR prompt templates.
"""

DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE = (
    "translate the speech from {source_language} into {target_language}. "
    "Return only the {target_language} translation, not the original {source_language} transcript."
)
