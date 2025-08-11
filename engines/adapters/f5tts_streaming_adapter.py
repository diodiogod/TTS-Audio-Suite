"""
F5-TTS Streaming Adapter

Enables F5-TTS engine to work with the universal streaming system.
Handles F5-TTS-specific model switching and longer inference times.
"""

import torch
import os
import hashlib
from typing import Dict, Any, List, Optional
import sys

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.streaming.streaming_interface import StreamingEngineAdapter
from utils.streaming.streaming_types import StreamingSegment, StreamingResult


class F5TTSStreamingAdapter(StreamingEngineAdapter):
    """
    Streaming adapter for F5-TTS engine.
    
    Enables F5-TTS to work with the universal streaming system,
    handling language-specific models (F5-DE, F5-FR, etc.) and
    F5-TTS's longer inference times.
    """
    
    def __init__(self, node_instance):
        """
        Initialize F5-TTS streaming adapter.
        
        Args:
            node_instance: F5TTSNode or F5TTSSRTNode instance
        """
        super().__init__(node_instance)
        
        # Cache for loaded F5-TTS models
        self._loaded_models = {}
        self._current_model = None
        self._current_language = None
        
        # F5-TTS specific sample rate
        self.sample_rate = 24000  # F5-TTS uses 24kHz
    
    def _get_engine_name(self) -> str:
        """Return the name of this engine."""
        return "f5tts"
    
    def _get_supported_languages(self) -> List[str]:
        """Return list of language codes supported by F5-TTS."""
        try:
            from utils.models.language_mapper import f5tts_language_mapper
            
            # Get supported language codes from mapper
            supported_codes = f5tts_language_mapper.get_supported_languages()
            
            # F5-TTS also supports local models
            if hasattr(self.node, 'get_available_models'):
                models = self.node.get_available_models()
                for model in models:
                    if model.startswith('local:'):
                        supported_codes.append(model)
            
            return supported_codes
            
        except ImportError:
            # Fallback to known F5-TTS languages
            return ['en', 'de', 'es', 'fr', 'it', 'jp', 'th', 'pt', 'pt-br', 'hi']
    
    def process_segment(self, segment: StreamingSegment, **kwargs) -> StreamingResult:
        """
        Process a single streaming segment using F5-TTS.
        
        Handles F5-TTS's specific requirements including reference audio,
        model switching, and longer inference times.
        
        Args:
            segment: StreamingSegment to process
            **kwargs: Processing parameters specific to F5-TTS
            
        Returns:
            StreamingResult with generated audio
        """
        start_time = self._get_current_time()
        
        try:
            # Ensure correct model is loaded for this language
            if self._current_language != segment.language:
                self.load_model_for_language(segment.language, kwargs.get('device', 'auto'))
            
            # Generate audio using F5-TTS
            audio = self._generate_with_f5tts(
                text=segment.text,
                voice_path=segment.voice_path,
                character=segment.character,
                language=segment.language,
                **kwargs
            )
            
            # Calculate duration (F5-TTS uses 24kHz)
            duration = float(audio.shape[-1]) / self.sample_rate if audio is not None else 0.0
            
            # Create successful result
            return StreamingResult(
                index=segment.index,
                audio=audio,
                duration=duration,
                character=segment.character,
                language=segment.language,
                processing_time=self._get_current_time() - start_time,
                worker_id=0,  # Will be set by worker
                success=True,
                metadata=segment.metadata
            )
            
        except Exception as e:
            # Return error result
            return StreamingResult(
                index=segment.index,
                audio=torch.zeros(1, 1000),
                duration=0.0,
                character=segment.character,
                language=segment.language,
                processing_time=self._get_current_time() - start_time,
                worker_id=0,
                success=False,
                error_msg=str(e),
                metadata=segment.metadata
            )
    
    def load_model_for_language(self, language: str, device: str = "auto") -> bool:
        """
        Load or switch to the F5-TTS model for specified language.
        
        F5-TTS has different models for different languages:
        - F5TTS_Base/F5TTS_v1_Base for English
        - F5-DE for German
        - F5-FR for French
        - etc.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'fr')
            device: Device to load model on
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Get the model name for this language
            model_name = self.get_model_for_language(language)
            
            # Check if we already have this model loaded
            if self._current_model == model_name and self._current_language == language:
                return True
            
            # Use universal smart model loader
            from utils.models.smart_loader import smart_model_loader
            
            def f5tts_load_callback(device: str, model: str):
                """Callback for F5-TTS model loading"""
                if hasattr(self.node, 'load_f5tts_model'):
                    # Use node's existing model loading (which now uses smart loader internally)
                    return self.node.load_f5tts_model(model, device)
                else:
                    # Fallback: try to load directly
                    from engines.f5tts.f5tts import F5TTSEngine
                    engine = F5TTSEngine()
                    engine.load_model(model, device)
                    return engine
            
            # Use smart loader to get or load the model
            model_instance, was_cached = smart_model_loader.load_model_if_needed(
                engine_type="f5tts",
                model_name=model_name,
                current_model=getattr(self.node, 'f5tts_model', None),
                device=device,
                load_callback=f5tts_load_callback
            )
            
            # Update node's model reference and our tracking
            if hasattr(self.node, 'f5tts_model'):
                self.node.f5tts_model = model_instance
            elif hasattr(self.node, 'f5tts'):
                self.node.f5tts = model_instance
            else:
                self.node.f5tts = model_instance
            
            # Cache the model for local reuse and update tracking
            self._loaded_models[model_name] = model_instance
            self._current_model = model_name
            self._current_language = language
            
            if was_cached:
                print(f"♻️ F5TTS STREAMING: Reused {model_name} from smart loader")
            else:
                print(f"✅ F5TTS STREAMING: Loaded {model_name} via smart loader")
            
            return True
                
        except Exception as e:
            print(f"❌ Failed to load F5-TTS model for {language}: {e}")
            return False
    
    def get_model_for_language(self, language: str) -> str:
        """
        Get the F5-TTS model name for specified language.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'fr')
            
        Returns:
            F5-TTS model name for this language
        """
        try:
            from utils.models.language_mapper import get_model_for_language
            
            # Handle local models
            if language.startswith('local:'):
                return language
            
            # Get model name from mapper
            default_model = getattr(self.node, 'default_model', 'F5TTS_v1_Base')
            return get_model_for_language('f5tts', language, default_model)
            
        except ImportError:
            # Fallback mapping
            language_map = {
                'en': 'F5TTS_v1_Base',
                'de': 'F5-DE',
                'es': 'F5-ES',
                'fr': 'F5-FR',
                'it': 'F5-IT',
                'jp': 'F5-JP',
                'th': 'F5-TH',
                'pt': 'F5-PT-BR',
                'pt-br': 'F5-PT-BR',
                'hi': 'F5-Hindi-Small'
            }
            return language_map.get(language, 'F5TTS_v1_Base')
    
    def preload_models(self, languages: List[str], device: str = "auto") -> Dict[str, bool]:
        """
        Pre-load F5-TTS models for multiple languages.
        
        F5-TTS models are larger and take longer to load, so pre-loading
        is especially beneficial for streaming performance.
        
        Args:
            languages: List of language codes to preload
            device: Device to load models on
            
        Returns:
            Dict mapping language -> success status
        """
        results = {}
        
        print(f"🚀 Pre-loading F5-TTS models for {len(languages)} languages...")
        
        for language in languages:
            try:
                # Load model for this language
                success = self.load_model_for_language(language, device)
                results[language] = success
                
                if success:
                    model_name = self.get_model_for_language(language)
                    print(f"✅ Pre-loaded F5-TTS model {model_name} for {language}")
                else:
                    print(f"❌ Failed to pre-load F5-TTS model for {language}")
                    
            except Exception as e:
                print(f"❌ Error pre-loading F5-TTS model for {language}: {e}")
                results[language] = False
        
        return results
    
    def _generate_with_f5tts(self, text: str, voice_path: str, character: str,
                              language: str, **kwargs) -> torch.Tensor:
        """
        Generate audio using F5-TTS model.
        
        Args:
            text: Text to generate
            voice_path: Path to reference audio file
            character: Character name
            language: Language code
            **kwargs: Additional F5-TTS parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract F5-TTS specific parameters
        speed = kwargs.get('speed', 0.9)
        sway_sampling_coef = kwargs.get('sway_sampling_coef', -1.0)
        cfg_strength = kwargs.get('cfg_strength', 2.0)
        seed = kwargs.get('seed', -1)
        target_rms = kwargs.get('target_rms', 0.1)
        
        # Load reference audio if available
        reference_audio = None
        reference_text = None
        
        if voice_path and voice_path != 'none' and os.path.exists(voice_path):
            # Load reference audio
            if hasattr(self.node, '_load_reference_audio'):
                reference_audio, reference_text = self.node._load_reference_audio(voice_path)
            else:
                # Simple loading fallback
                import torchaudio
                reference_audio, _ = torchaudio.load(voice_path)
                
                # Check for text file
                text_path = voice_path.replace('.wav', '.txt').replace('.mp3', '.txt')
                if os.path.exists(text_path):
                    with open(text_path, 'r', encoding='utf-8') as f:
                        reference_text = f.read().strip()
        
        # Use reference text if no reference audio text found
        if reference_audio is not None and not reference_text:
            reference_text = "This is a reference voice for synthesis."
        
        # Generate with F5-TTS
        if hasattr(self.node, 'f5tts') and self.node.f5tts:
            # Use node's F5-TTS instance
            audio = self.node.f5tts.generate(
                text=text,
                ref_audio=reference_audio,
                ref_text=reference_text,
                speed=speed,
                sway_sampling_coef=sway_sampling_coef,
                cfg_strength=cfg_strength,
                seed=seed if seed >= 0 else None,
                target_rms=target_rms
            )
            
            # Apply caching if enabled
            if kwargs.get('enable_audio_cache', True):
                cache_key = self._get_cache_key(text, character, language, kwargs)
                self._cache_audio(cache_key, audio)
            
            return audio
            
        else:
            raise RuntimeError("F5-TTS model not loaded")
    
    def _get_cache_key(self, text: str, character: str, language: str, kwargs: Dict) -> str:
        """Generate cache key for audio caching."""
        # Create unique key from parameters
        key_parts = [
            text,
            character,
            language,
            str(kwargs.get('speed', 0.9)),
            str(kwargs.get('cfg_strength', 2.0)),
            str(kwargs.get('seed', -1))
        ]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_audio(self, cache_key: str, audio: torch.Tensor):
        """Cache generated audio."""
        # Use node's caching if available
        if hasattr(self.node, 'audio_cache'):
            self.node.audio_cache[cache_key] = audio.clone()
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        import time
        return time.time()
    
    def validate_segment(self, segment: StreamingSegment) -> bool:
        """
        Validate that F5-TTS can process this segment.
        
        Args:
            segment: Segment to validate
            
        Returns:
            True if segment can be processed
        """
        # Check language support
        if segment.language not in self.supported_languages:
            if not segment.language.startswith('local:'):
                print(f"❌ F5-TTS doesn't support language: {segment.language}")
                return False
        
        # F5-TTS requires non-empty text
        if not segment.text.strip():
            print(f"❌ F5-TTS requires non-empty text for segment {segment.index}")
            return False
        
        return True