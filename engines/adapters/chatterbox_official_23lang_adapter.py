"""
ChatterBox Official 23-Lang Engine Adapter - Engine-specific adapter for ChatterBox Official 23-Lang TTS
Provides standardized interface for ChatterBox Official 23-Lang operations in multilingual engine
With native batch processing support for parallel text generation
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import concurrent.futures
import threading
# Use absolute import to avoid relative import issues in ComfyUI
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.models.language_mapper import get_model_for_language


class ChatterBoxOfficial23LangEngineAdapter:
    """Engine-specific adapter for ChatterBox TTS with batch processing support."""
    
    def __init__(self, node_instance):
        """
        Initialize ChatterBox adapter with batch processing capabilities.
        
        Args:
            node_instance: ChatterboxTTSNode or SRTTTSNode instance
        """
        self.node = node_instance
        self.engine_type = "chatterbox_official_23lang"
        self.batch_size = 4  # Default batch size for parallel processing
        self.enable_batch_processing = True  # Enable batch processing by default
        self._generation_lock = threading.Lock()  # Thread safety for model access
    
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
        # Extract ChatterBox Official 23-Lang specific parameters
        exaggeration = params.get("exaggeration", 1.0)
        temperature = params.get("temperature", 0.8)
        cfg_weight = params.get("cfg_weight", 1.0)
        repetition_penalty = params.get("repetition_penalty", 2.0)
        min_p = params.get("min_p", 0.05)
        top_p = params.get("top_p", 1.0)
        language_id = params.get("language_id", "en")
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
                engine_type="chatterbox_official_23lang",
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
        
        # CRITICAL FIX: Handle caching based on pause tag presence
        # If pause tags detected, let internal pause processor handle caching of text segments
        # If no pause tags, use external caching as normal
        enable_pause_tags = params.get("enable_pause_tags", True)
        has_pause_tags = enable_pause_tags and "[pause:" in text
        
        if has_pause_tags:
            # Let internal pause processor handle caching of individual text segments
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
                enable_cache=True,  # Enable internal caching for pause tag processing
                crash_protection_template=params.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=params.get("stable_audio_component", "main_reference")
            )
            # Don't use external cache for pause tag segments
        else:
            # No pause tags, use external caching as normal
            audio_result = self.node._generate_tts_with_pause_tags(
                text=text,
                audio_prompt=char_audio,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                language=params.get("current_language", params.get("model", "English")),
                enable_pause_tags=False,
                character=character,
                seed=seed,
                enable_cache=False,  # Disable internal caching since we handle it externally
                crash_protection_template=params.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=params.get("stable_audio_component", "main_reference")
            )
            
            # Cache the result if external caching is enabled
            if cache_fn:
                cache_fn(text, audio_result)
        
        return audio_result
    
    def generate_batch_audio(self, text_segments: List[str], char_audio: str,
                            character: str = "narrator", batch_size: Optional[int] = None,
                            **params) -> List[torch.Tensor]:
        """
        Generate ChatterBox audio for multiple text segments using true batch processing.
        
        Args:
            text_segments: List of text segments to generate audio for
            char_audio: Reference audio file path
            character: Character name for caching
            batch_size: Optional batch size override (uses self.batch_size if None)
            **params: Additional ChatterBox parameters
            
        Returns:
            List of generated audio tensors
        """
        if not self.enable_batch_processing or len(text_segments) == 1:
            # Fall back to sequential processing if batch processing is disabled or single segment
            return [self.generate_segment_audio(text, char_audio, character, **params)
                   for text in text_segments]
        
        batch_size = batch_size or self.batch_size
        print(f"🚀 ChatterBox batch processing: {len(text_segments)} segments in batches of {batch_size}")
        
        audio_results = []
        
        # Process segments in batches using the engine's generate_batch method
        for i in range(0, len(text_segments), batch_size):
            batch_texts = text_segments[i:i + batch_size]
            print(f"🔄 Processing batch {i//batch_size + 1}: {len(batch_texts)} segments")
            
            try:
                # Extract ChatterBox specific parameters
                exaggeration = params.get("exaggeration", 1.0)
                temperature = params.get("temperature", 0.8)
                cfg_weight = params.get("cfg_weight", 1.0)
                
                # Use the ChatterBox engine's generate_batch method for true batch processing
                if hasattr(self.node, 'tts_model') and hasattr(self.node.tts_model, 'generate_batch'):
                    print(f"🔧 Using overlapping batch processing with max_workers={self.batch_size}")
                    batch_audio = self.node.tts_model.generate_batch(
                        texts=batch_texts,
                        audio_prompt_path=char_audio,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        batch_size=len(batch_texts),
                        max_workers=self.batch_size  # Use the UI batch_size as max_workers
                    )
                    
                    print(f"✅ Batch {i//batch_size + 1} completed: {len(batch_audio)} audio segments generated")
                    audio_results.extend(batch_audio)
                    
                else:
                    # Fallback to sequential generation if batch method not available
                    print(f"⚠️ Batch method not available, falling back to sequential processing")
                    for text in batch_texts:
                        audio = self.generate_segment_audio(text, char_audio, character, **params)
                        audio_results.append(audio)
                        
            except Exception as e:
                print(f"⚠️ Batch processing failed for batch {i//batch_size + 1}: {e}")
                print(f"🔄 Falling back to sequential processing for this batch")
                # Fall back to sequential generation for failed batch
                for text in batch_texts:
                    try:
                        audio = self.generate_segment_audio(text, char_audio, character, **params)
                        audio_results.append(audio)
                    except Exception as fallback_error:
                        print(f"❌ Sequential fallback also failed for '{text[:30]}...': {fallback_error}")
                        # Generate silence as last resort
                        from utils.audio.processing import AudioProcessingUtils
                        silence = AudioProcessingUtils.create_silence(1.0, 44100)
                        audio_results.append(silence)
        
        print(f"🎉 Batch processing complete: {len(audio_results)} total segments generated")
        return audio_results
    
    def _generate_single_with_lock(self, text: str, char_audio: str,
                                  character: str = "narrator", **params) -> torch.Tensor:
        """
        Generate audio for a single segment with thread-safe model access.
        
        Args:
            text: Text to generate audio for
            char_audio: Reference audio file path
            character: Character name
            **params: Additional parameters
            
        Returns:
            Generated audio tensor
        """
        # Use lock to ensure thread-safe model access
        with self._generation_lock:
            return self.generate_segment_audio(text, char_audio, character, **params)
    
    def set_batch_processing(self, enabled: bool, batch_size: Optional[int] = None):
        """
        Configure batch processing settings.
        
        Args:
            enabled: Whether to enable batch processing
            batch_size: Optional new batch size (1-32 supported, uses overlapping parallel processing)
        """
        self.enable_batch_processing = enabled
        if batch_size is not None:
            self.batch_size = max(1, min(32, batch_size))  # Allow up to 32 parallel workers
        
        status = "enabled" if enabled else "disabled"
        print(f"🔧 ChatterBox batch processing {status} (batch_size={self.batch_size})")
    
    def combine_audio_chunks(self, audio_segments: List[torch.Tensor], **params) -> torch.Tensor:
        """
        Combine ChatterBox audio segments using modular combination utility.
        
        Args:
            audio_segments: List of audio tensors to combine
            **params: Combination parameters
            
        Returns:
            Combined audio tensor, or tuple with timing info if requested
        """
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Use modular chunk combiner with ChatterBox settings
        from utils.audio.chunk_combiner import ChunkCombiner
        
        method = params.get("combination_method", "auto")
        silence_ms = params.get("silence_ms", 100)
        text_length = params.get("text_length", 0)
        original_text = params.get("original_text", "")
        text_chunks = params.get("text_chunks", None)
        return_info = params.get("return_timing_info", False)
        
        print(f"🔗 Combining {len(audio_segments)} ChatterBox chunks using '{method}' method")
        
        return ChunkCombiner.combine_chunks(
            audio_segments=audio_segments,
            method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=44100,  # ChatterBox sample rate
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
            return_info=return_info
        )
    
    def generate_audio(self, text: str, voice_preset: str, voice_settings: dict,
                      output_filename: str, language: str = "auto",
                      enable_batch_processing: bool = False, batch_size: int = 4) -> dict:
        """
        Generate audio using ChatterBox adapter with batch processing support.
        
        This method provides the interface expected by the unified TTS system.
        """
        # For single text generation, use the existing segment generation
        if not enable_batch_processing:
            print(f"🎤 ChatterBox single generation: '{text[:50]}...'")
            
            # Use the existing node functionality for single generation
            audio_result = self.node.generate_speech(
                text=text,
                language=language,
                device=self.node.device if hasattr(self.node, 'device') else 'auto',
                exaggeration=voice_settings.get('exaggeration', 0.5),
                temperature=voice_settings.get('temperature', 0.8),
                cfg_weight=voice_settings.get('cfg_weight', 0.5),
                seed=voice_settings.get('seed', 0),
                reference_audio=voice_settings.get('reference_audio'),
                audio_prompt_path=voice_settings.get('audio_prompt_path', ''),
                batch_size=1 if not enable_batch_processing else batch_size
            )
            
            return {
                'audio': audio_result[0],  # Audio tensor
                'info': audio_result[1] if len(audio_result) > 1 else 'Generated with ChatterBox'
            }
        else:
            # For batch processing, split text into chunks and use batch generation
            print(f"🚀 ChatterBox batch generation enabled: batch_size={batch_size}")
            
            # Split text into reasonable chunks for batch processing
            from utils.text.chunking import ImprovedChatterBoxChunker
            chunker = ImprovedChatterBoxChunker()
            
            # Use chunking to split text into segments
            max_chars_per_chunk = voice_settings.get('max_chars_per_chunk', 400)
            if len(text) > max_chars_per_chunk:
                text_segments = chunker.split_into_chunks(text, max_chars_per_chunk)
                print(f"📝 Text split into {len(text_segments)} chunks for batch processing")
            else:
                text_segments = [text]
            
            # Generate audio for all segments using batch processing
            audio_segments = self.generate_batch_audio(
                text_segments=text_segments,
                char_audio=voice_settings.get('audio_prompt_path', ''),
                character=voice_preset or 'narrator',
                batch_size=batch_size,
                exaggeration=voice_settings.get('exaggeration', 0.5),
                temperature=voice_settings.get('temperature', 0.8),
                cfg_weight=voice_settings.get('cfg_weight', 0.5),
                seed=voice_settings.get('seed', 0),
                current_language=language,
                device=self.node.device if hasattr(self.node, 'device') else 'auto'
            )
            
            # Combine audio segments
            if len(audio_segments) == 1:
                combined_audio = audio_segments[0]
            else:
                combined_audio = self.combine_audio_chunks(audio_segments)
            
            return {
                'audio': combined_audio,
                'info': f'Generated {len(text_segments)} chunks using ChatterBox batch processing (batch_size={batch_size})'
            }

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