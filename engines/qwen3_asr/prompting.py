"""
Shared Qwen3 ASR prompt/context templates.
"""

DEFAULT_TRANSLATE_INSTRUCTION_TEMPLATE = (
    "Translate the speech from {source_language} into {target_language} text. "
    "Return only the translated text."
)
