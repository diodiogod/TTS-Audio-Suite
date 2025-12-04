"""
Configuration module for Step-Audio
"""

from .prompts import AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL, AUDIO_EDIT_SYSTEM_PROMPT
from .edit_config import get_supported_edit_types

__all__ = [
    'AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL',
    'AUDIO_EDIT_SYSTEM_PROMPT',
    'get_supported_edit_types'
]