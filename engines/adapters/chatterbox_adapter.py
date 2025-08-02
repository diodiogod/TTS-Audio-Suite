"""
ChatterBox Engine Adapter - Engine-specific adapter for ChatterBox TTS
Provides standardized interface for ChatterBox operations in multilingual engine
"""

import torch
from typing import Dict, Any, Optional, List
# Use absolute import to avoid relative import issues in ComfyUI
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.models.language_mapper import get_model_for_language


class ChatterBoxEngineAdapter:
    """Engine-specific adapter for ChatterBox TTS."""
    
    def __init__(self, node_instance):
        """
        Initialize ChatterBox adapter.
        
        Args:
            node_instance: ChatterboxTTSNode or SRTTTSNode instance
        """
        self.node = node_instance
        self.engine_type = "chatterbox"
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """
        Get ChatterBox model name for specified language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'de', 'no')
            default_model: Default model name (language)
            
        Returns:
            ChatterBox model name (language) for the specified language code
        """
        return get_model_for_language(self.engine_type, lang_code, default_model)
    
    def load_base_model(self, language: str, device: str):
        """
        Load base ChatterBox model.
        
        Args:
            language: Language model to load (e.g., "English", "German")
            device: Device to load model on
        """
        self.node.load_tts_model(device, language)
    
    def load_language_model(self, language: str, device: str):
        """
        Load language-specific ChatterBox model.
        
        Args:
            language: Language model to load (e.g., "German", "Norwegian")
            device: Device to load model on
        """
        self.node.load_tts_model(device, language)
    
    def generate_segment_audio(self, text: str, char_audio: str, 
                             character: str = "narrator", **params) -> torch.Tensor:
        """
        Generate ChatterBox audio for a text segment.
        
        Args:
            text: Text to generate audio for
            char_audio: Reference audio file path
            character: Character name for caching
            **params: Additional ChatterBox parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract ChatterBox specific parameters
        exaggeration = params.get("exaggeration", 1.0)
        temperature = params.get("temperature", 0.8)
        cfg_weight = params.get("cfg_weight", 1.0)
        seed = params.get("seed", 0)
        enable_cache = params.get("enable_audio_cache", True)
        
        # Create cache function if caching is enabled
        cache_fn = None
        if enable_cache:
            from utils.audio.cache import create_cache_function
            
            # Get current language/model for cache key
            current_language = params.get("current_language", params.get("model", "English"))
            audio_component = params.get("stable_audio_component", "main_reference")
            if character != "narrator":
                audio_component = f"char_file_{character}"
            
            # Get model source
            model_source = params.get("model_source")
            if not model_source and hasattr(self.node, 'model_manager'):
                model_source = self.node.model_manager.get_model_source("tts")
            
            cache_fn = create_cache_function(
                engine_type="chatterbox",
                character=character,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                seed=seed,
                audio_component=audio_component,
                model_source=model_source or "unknown",
                device=params.get("device", "auto"),
                language=current_language
            )
        
        # Handle caching externally for consistency with F5-TTS
        if cache_fn:
            # Check cache first
            cached_audio = cache_fn(text)
            if cached_audio is not None:
                return cached_audio
        
        # Generate audio using ChatterBox with pause tag support
        audio_result = self.node._generate_tts_with_pause_tags(
            text=text,
            audio_prompt=char_audio,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            language=params.get("current_language", params.get("model", "English")),
            enable_pause_tags=True,
            character=character,
            seed=seed,
            enable_cache=False,  # Disable internal caching since we handle it externally
            crash_protection_template=params.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
            stable_audio_component=params.get("stable_audio_component", "main_reference")
        )
        
        # Cache the result if caching is enabled
        if cache_fn:
            cache_fn(text, audio_result)
        
        return audio_result
    
    def combine_audio_chunks(self, audio_segments: List[torch.Tensor], **params) -> torch.Tensor:
        """
        Combine ChatterBox audio segments.
        
        Args:
            audio_segments: List of audio tensors to combine
            **params: Combination parameters
            
        Returns:
            Combined audio tensor
        """
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # ChatterBox uses simple concatenation
        from utils.audio.processing import AudioProcessingUtils
        return AudioProcessingUtils.concatenate_audio_segments(audio_segments, "simple")
    
    def _get_audio_duration(self, audio_tensor: torch.Tensor) -> float:
        """Calculate audio duration in seconds."""
        # ChatterBox uses 44.1kHz sample rate
        if audio_tensor.dim() == 1:
            num_samples = audio_tensor.shape[0]
        elif audio_tensor.dim() == 2:
            num_samples = audio_tensor.shape[1]
        else:
            num_samples = audio_tensor.numel()
        
        return num_samples / 44100  # ChatterBox sample rate