"""
F5-TTS Engine Adapter - Engine-specific adapter for F5-TTS
Provides standardized interface for F5-TTS operations in multilingual engine
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


class F5TTSEngineAdapter:
    """Engine-specific adapter for F5-TTS."""
    
    def __init__(self, node_instance):
        """
        Initialize F5-TTS adapter.
        
        Args:
            node_instance: F5TTSNode or F5TTSSRTNode instance
        """
        self.node = node_instance
        self.engine_type = "f5tts"
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """
        Get F5-TTS model name for specified language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'de', 'fr')
            default_model: Default model name
            
        Returns:
            F5-TTS model name for the language
        """
        return get_model_for_language(self.engine_type, lang_code, default_model)
    
    def load_base_model(self, model_name: str, device: str):
        """
        Load base F5-TTS model.
        
        Args:
            model_name: Model name to load
            device: Device to load model on
        """
        # Check if the model is already loaded to avoid redundant loading
        current_model = getattr(self.node, 'current_model_name', None)
        if current_model == model_name:
            print(f"💾 F5-TTS adapter: Model '{model_name}' already loaded - skipping base model load")
            return
        
        self.node.load_f5tts_model(model_name, device)
    
    def load_language_model(self, model_name: str, device: str):
        """
        Load language-specific F5-TTS model.
        
        Args:
            model_name: Language-specific model name
            device: Device to load model on
        """
        # Check if the model is already loaded to avoid redundant loading
        current_model = getattr(self.node, 'current_model_name', None)
        if current_model == model_name:
            print(f"💾 F5-TTS adapter: Model '{model_name}' already loaded - skipping language model load")
            return
        
        self.node.load_f5tts_model(model_name, device)
    
    def generate_segment_audio(self, text: str, char_audio: str, char_text: str, 
                             character: str = "narrator", **params) -> torch.Tensor:
        """
        Generate F5-TTS audio for a text segment.
        
        Args:
            text: Text to generate audio for
            char_audio: Reference audio file path
            char_text: Reference text
            character: Character name for caching
            **params: Additional F5-TTS parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract F5-TTS specific parameters
        temperature = params.get("temperature", 0.8)
        speed = params.get("speed", 1.0)
        target_rms = params.get("target_rms", 0.1)
        cross_fade_duration = params.get("cross_fade_duration", 0.15)
        nfe_step = params.get("nfe_step", 32)
        cfg_strength = params.get("cfg_strength", 2.0)
        seed = params.get("seed", 0)
        enable_cache = params.get("enable_audio_cache", True)
        
        # Validate and clamp nfe_step to prevent ODE solver issues
        safe_nfe_step = max(1, min(nfe_step, 71))
        if safe_nfe_step != nfe_step:
            print(f"⚠️ F5-TTS: Clamped nfe_step from {nfe_step} to {safe_nfe_step} to prevent ODE solver issues")
        
        # Create cache function if caching is enabled
        cache_fn = None
        if enable_cache:
            from utils.audio.cache import create_cache_function
            
            # Get audio component for cache key
            audio_component = params.get("stable_audio_component", "main_reference")
            if character != "narrator":
                audio_component = f"char_file_{character}"
            
            # Get current model name for cache key
            current_model = getattr(self.node, 'current_model_name', params.get("model", "F5TTS_Base"))
            
            cache_fn = create_cache_function(
                engine_type="f5tts",
                character=character,
                model_name=current_model,
                device=params.get("device", "auto"),
                audio_component=audio_component,
                ref_text=char_text,
                temperature=temperature,
                speed=speed,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=safe_nfe_step,
                cfg_strength=cfg_strength,
                seed=seed
            )
        
        # Generate audio using F5-TTS with pause tag support
        return self.node.generate_f5tts_with_pause_tags(
            text=text,
            ref_audio_path=char_audio,
            ref_text=char_text,
            enable_pause_tags=True,
            character=character,
            seed=seed,
            enable_cache=enable_cache,
            cache_fn=cache_fn,
            temperature=temperature,
            speed=speed,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=safe_nfe_step,
            cfg_strength=cfg_strength
        )
    
    def combine_audio_chunks(self, audio_segments: List[torch.Tensor], **params) -> torch.Tensor:
        """
        Combine F5-TTS audio segments using modular combination utility.
        
        Args:
            audio_segments: List of audio tensors to combine
            **params: Combination parameters
            
        Returns:
            Combined audio tensor, or tuple with timing info if requested
        """
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        method = params.get("combination_method", "auto")
        silence_ms = params.get("silence_ms", 100)
        text_length = params.get("text_length", 0)
        original_text = params.get("original_text", "")
        text_chunks = params.get("text_chunks", None)
        return_info = params.get("return_timing_info", False)
        
        return self.node.combine_f5tts_audio_chunks(
            audio_segments, method, silence_ms, text_length,
            original_text=original_text, text_chunks=text_chunks, return_info=return_info
        )
    
    def _get_audio_duration(self, audio_tensor: torch.Tensor) -> float:
        """Calculate audio duration in seconds."""
        if hasattr(self.node, 'AudioTimingUtils'):
            return self.node.AudioTimingUtils.get_audio_duration(audio_tensor, self.node.f5tts_sample_rate)
        else:
            # Fallback calculation
            if audio_tensor.dim() == 1:
                num_samples = audio_tensor.shape[0]
            elif audio_tensor.dim() == 2:
                num_samples = audio_tensor.shape[1]
            else:
                num_samples = audio_tensor.numel()
            
            return num_samples / 24000  # F5-TTS sample rate