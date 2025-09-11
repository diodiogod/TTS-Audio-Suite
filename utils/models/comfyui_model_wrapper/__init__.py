"""
ComfyUI Model Wrapper for TTS Audio Suite

This module provides wrappers that make TTS models compatible with ComfyUI's
model management system, enabling automatic memory management, "Clear VRAM" 
button functionality, and proper integration with ComfyUI's model lifecycle.
"""

# Core exports for backward compatibility
from .base_wrapper import ComfyUIModelWrapper, ModelInfo
from .model_manager import ComfyUITTSModelManager, tts_model_manager
from .cache_utils import is_engine_cache_valid, _global_cache_invalidation_flag

__all__ = [
    'ComfyUIModelWrapper',
    'ModelInfo', 
    'ComfyUITTSModelManager',
    'tts_model_manager',
    'is_engine_cache_valid',
    '_global_cache_invalidation_flag'
]