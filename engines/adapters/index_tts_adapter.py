"""
IndexTTS-2 Engine Adapter

Provides standardized interface for IndexTTS-2 integration with TTS Audio Suite.
Handles parameter mapping, character switching, and emotion control.
"""

import os
import torch
from typing import Dict, Any, Optional, List, Union

from engines.index_tts.index_tts import IndexTTSEngine
from engines.index_tts.index_tts_downloader import index_tts_downloader
from utils.text.character_parser import character_parser
from utils.voice.discovery import get_character_mapping, get_available_characters
from utils.audio.cache import get_audio_cache


class IndexTTSAdapter:
    """
    Adapter for IndexTTS-2 engine providing unified interface compatibility.
    
    Handles:
    - Parameter mapping between unified interface and IndexTTS-2
    - Character switching with [character:emotion_ref] syntax 
    - Emotion control via audio references or vectors
    - Caching integration
    - Model management
    """
    
    def __init__(self):
        """Initialize the IndexTTS-2 adapter."""
        self.engine = None
        self.audio_cache = get_audio_cache()
    
    def initialize_engine(self, 
                         model_path: Optional[str] = None,
                         device: str = "auto",
                         use_fp16: bool = True,
                         use_cuda_kernel: Optional[bool] = None,
                         use_deepspeed: bool = False):
        """
        Initialize IndexTTS-2 engine.
        
        Args:
            model_path: Path to model directory (auto-downloaded if None)
            device: Target device
            use_fp16: Use FP16 for inference
            use_cuda_kernel: Use BigVGAN CUDA kernels
            use_deepspeed: Use DeepSpeed optimization
        """
        # Auto-download model if not provided or if "auto-download" is specified
        if model_path is None or model_path == "auto-download":
            if not index_tts_downloader.is_model_available():
                print("📥 IndexTTS-2 model not found, downloading...")
                model_path = index_tts_downloader.download_model()
            else:
                model_path = index_tts_downloader.get_model_path()
                
        # Initialize engine
        self.engine = IndexTTSEngine(
            model_dir=model_path,
            device=device,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )
        
    
    def generate(self,
                text: str,
                speaker_audio: Optional[str] = None,
                emotion_audio: Optional[str] = None,
                emotion_alpha: float = 1.0,
                emotion_vector: Optional[List[float]] = None,
                use_emotion_text: bool = False,
                emotion_text: Optional[str] = None,
                use_random: bool = False,
                interval_silence: int = 200,
                max_text_tokens_per_segment: int = 120,
                # Generation parameters
                temperature: float = 0.8,
                top_p: float = 0.8,
                top_k: int = 30,
                length_penalty: float = 0.0,
                num_beams: int = 3,
                repetition_penalty: float = 10.0,
                max_mel_tokens: int = 1500,
                **kwargs) -> torch.Tensor:
        """
        Generate speech with IndexTTS-2.
        
        Args:
            text: Text to synthesize (supports [character:emotion] tags)
            speaker_audio: Speaker reference audio file path
            emotion_audio: Emotion reference audio file path
            emotion_alpha: Emotion blend factor (0.0-1.0)
            emotion_vector: Manual emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            use_emotion_text: Extract emotions from text
            emotion_text: Custom emotion description text
            use_random: Enable random sampling
            interval_silence: Silence between segments (ms)
            max_text_tokens_per_segment: Max tokens per segment
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            length_penalty: Length penalty for beam search
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty
            max_mel_tokens: Maximum mel tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated audio tensor [1, samples] at 22050 Hz
        """
        if self.engine is None:
            self.initialize_engine()

        # Check if text actually contains character tags before processing
        has_character_tags = character_parser.CHARACTER_TAG_PATTERN.search(text) is not None

        if has_character_tags:
            # Parse character switching tags with emotion support
            processed_segments = self._process_character_tags_with_emotions(text)
            
            if len(processed_segments) > 1:
                # Multi-segment character switching - process each segment separately
                return self._generate_multi_character_segments(processed_segments, speaker_audio, emotion_audio, **kwargs)
            elif processed_segments:
                # Single character segment
                first_segment = processed_segments[0]
                processed_text = first_segment.get('text', '').strip()
                character_name = first_segment.get('character')
                emotion_ref = first_segment.get('emotion')
            else:
                processed_text = text
                character_name = None
                emotion_ref = None
        else:
            # No character tags - use text as is
            processed_text = text
            character_name = None
            emotion_ref = None
        
        # Determine final speaker and emotion audio
        final_speaker_audio = speaker_audio

        # Handle Character Voices emotion_audio format
        if emotion_audio and isinstance(emotion_audio, dict):
            if "audio_path" in emotion_audio:
                # Character Voices format: {'audio': {...}, 'audio_path': 'path', ...}
                final_emotion_audio = emotion_audio["audio_path"]
                print(f"🎭 Using Character Voices emotion audio: {emotion_audio.get('character_name', 'unknown')} -> {final_emotion_audio}")
            elif "waveform" in emotion_audio:
                # Direct AUDIO format: {'waveform': tensor, 'sample_rate': rate}
                final_emotion_audio = emotion_audio
            else:
                final_emotion_audio = emotion_audio
        else:
            final_emotion_audio = emotion_audio
        
        # Only do character mapping if we actually have character tags
        if has_character_tags:
            # Collect all characters that might be needed
            all_characters = []
            if character_name:
                all_characters.append(character_name)
            if emotion_ref:
                all_characters.append(emotion_ref)
                
            # Get character mapping for IndexTTS
            if all_characters:
                character_mapping = get_character_mapping(all_characters, engine_type="index_tts")
            
            if character_name and character_name in character_mapping:
                # IndexTTS returns (audio_path, reference_text) tuples, we need audio_path
                character_audio_path = character_mapping[character_name][0]
                if character_audio_path is not None:
                    final_speaker_audio = character_audio_path
                    print(f"🎭 Using character voice: {character_name} -> {final_speaker_audio}")
                else:
                    print(f"⚠️ Character '{character_name}' mapping returned None, using original speaker audio: {final_speaker_audio}")
            elif character_name:
                print(f"⚠️ Character '{character_name}' not found in character mapping, using original speaker audio: {final_speaker_audio}")
                        
            if emotion_ref and emotion_ref in character_mapping:
                # IndexTTS returns (audio_path, reference_text) tuples, we need audio_path
                final_emotion_audio = character_mapping[emotion_ref][0]
                print(f"😊 Using emotion reference: {emotion_ref} -> {final_emotion_audio}")
                    
        # For caching, we need stable identifiers for audio references
        # Convert file paths to content hashes if they're temp files
        cache_speaker_audio = self._get_stable_audio_identifier(final_speaker_audio)
        cache_emotion_audio = self._get_stable_audio_identifier(final_emotion_audio) if final_emotion_audio else final_emotion_audio

        # Check cache before generation
        cache_key = self._generate_cache_key(
            text=processed_text,
            speaker_audio=cache_speaker_audio,
            emotion_audio=cache_emotion_audio,
            emotion_alpha=emotion_alpha,
            emotion_vector=emotion_vector,
            use_emotion_text=use_emotion_text,
            emotion_text=emotion_text,
            use_random=use_random,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            interval_silence=interval_silence,
            **kwargs  # Include seed and other kwargs in cache key
        )

        # print(f"🐛 IndexTTS-2 cache key params: text='{processed_text[:30]}...', seed={kwargs.get('seed', 'missing')}, speaker='{final_speaker_audio}', emotion='{final_emotion_audio}'")
        # print(f"🐛 IndexTTS-2 full cache key: {cache_key}")
        # print(f"🐛 IndexTTS-2 cache stats: {self.audio_cache.get_cache_stats()}")

        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            character_desc = character_name or 'narrator'
            print(f"💾 Using cached IndexTTS-2 audio for '{character_desc}': '{processed_text[:30]}...'")
            return cached_audio[0]
        # else:
        #     print(f"❌ IndexTTS-2 cache MISS for key: {cache_key}")
        #     # Show first few existing keys for comparison
        #     # Access the global cache properly
        #     from utils.audio.cache import GLOBAL_AUDIO_CACHE
        #     existing_keys = list(GLOBAL_AUDIO_CACHE.keys())
        #     if existing_keys:
        #         print(f"🔍 Existing cache keys (first 3): {existing_keys[:3]}")
        #     else:
        #         print(f"🔍 Cache is empty")
            
        # Handle seed for deterministic generation
        seed = kwargs.get('seed', 0)
        if seed != 0:
            torch.manual_seed(seed)
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

        # Filter out seed from kwargs (used for caching but not supported by IndexTTS engine)
        engine_kwargs = {k: v for k, v in kwargs.items() if k != 'seed'}

        # Apply consistent emotion priority: emotion_audio takes precedence over other emotion controls
        # This ensures consistent behavior whether using character tags or direct engine inputs
        if final_emotion_audio:
            # emotion_audio connected - disable other emotion controls
            final_emotion_vector = None
            final_use_emotion_text = False
            final_emotion_text = None
        else:
            # No emotion_audio - use provided emotion controls
            final_emotion_vector = emotion_vector
            final_use_emotion_text = use_emotion_text
            final_emotion_text = emotion_text

        # Generate audio with OOM protection
        try:
            audio = self.engine.generate(
                text=processed_text,
                speaker_audio=final_speaker_audio,
                emotion_audio=final_emotion_audio,
                emotion_alpha=emotion_alpha,
                emotion_vector=final_emotion_vector,
                use_emotion_text=final_use_emotion_text,
                emotion_text=final_emotion_text,
                use_random=use_random,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
                **engine_kwargs
            )
        except torch.OutOfMemoryError as e:
            # Analyze audio after OOM to provide helpful feedback
            audio_analysis = self._analyze_audio_after_oom(final_speaker_audio, final_emotion_audio, max_mel_tokens)
            raise RuntimeError(
                f"❌ IndexTTS-2 ran out of GPU memory during generation.\n{audio_analysis}"
            ) from e
        
        # Cache the result
        duration = audio.shape[-1] / 22050.0  # IndexTTS-2 uses 22050 Hz
        self.audio_cache.cache_audio(cache_key, audio, duration)
        # print(f"💾 IndexTTS-2 CACHED audio with key: {cache_key}")
        
        return audio
    
    def _process_character_tags_with_emotions(self, text: str) -> List[Dict[str, Any]]:
        """
        Process character switching tags with emotion support.
        """
        from utils.text.character_parser.base_parser import CharacterParser
        from utils.voice.discovery import get_available_characters, voice_discovery

        # Create a temporary parser and set it up properly
        temp_parser = CharacterParser()

        # Get actual available characters and aliases
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        # Build complete available set (like before but not hardcoded)
        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        temp_parser.set_available_characters(list(all_available))

        # Set language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            temp_parser.set_character_language_default(char, lang)

        # Use emotion-aware parsing
        segments = temp_parser.split_by_character_with_emotions(text)

        processed_segments = []
        for character, segment_text, language, emotion in segments:
            segment_info = {
                'character': character,
                'text': segment_text,
                'language': language,
                'emotion': emotion
            }
            processed_segments.append(segment_info)

        return processed_segments
    
    def _generate_multi_character_segments(self, segments: List[Dict[str, Any]], 
                                           default_speaker_audio: Optional[str], 
                                           default_emotion_audio: Optional[str], 
                                           **kwargs) -> torch.Tensor:
        """
        Generate audio for multiple character segments and combine them.
        
        Args:
            segments: List of segment dictionaries with character, text, language, emotion info
            default_speaker_audio: Default speaker reference audio
            default_emotion_audio: Default emotion reference audio  
            **kwargs: Additional generation parameters
            
        Returns:
            Combined audio tensor [1, samples] at 22050 Hz
        """
        audio_segments = []
        
        # Get character mapping for all unique characters
        unique_characters = set()
        for segment in segments:
            if segment.get('character'):
                unique_characters.add(segment['character'])
            if segment.get('emotion'):
                unique_characters.add(segment['emotion'])
        
        # Get character mapping for IndexTTS
        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="index_tts")
        
        print(f"🎭 IndexTTS-2: Processing {len(segments)} character segment(s) - {', '.join([s.get('character', 'narrator') for s in segments])}")
        
        for segment in segments:
            character_name = segment.get('character', 'narrator')
            segment_text = segment.get('text', '').strip()
            emotion_ref = segment.get('emotion')
            
            if not segment_text:
                continue
                
            # Determine speaker audio for this segment
            speaker_audio = default_speaker_audio
            if character_name and character_name in character_mapping:
                character_audio_path = character_mapping[character_name][0]
                if character_audio_path:
                    speaker_audio = character_audio_path
                    print(f"📖 Using character voice '{character_name}' | Ref: '{speaker_audio}'")
                else:
                    print(f"⚠️ Character '{character_name}' has no audio reference, using default")
            
            # Determine emotion audio for this segment
            emotion_audio = default_emotion_audio  
            if emotion_ref and emotion_ref in character_mapping:
                emotion_audio_path = character_mapping[emotion_ref][0]
                if emotion_audio_path:
                    emotion_audio = emotion_audio_path
                    print(f"😊 Using emotion reference '{emotion_ref}' | Ref: '{emotion_audio}'")
            
            # Generate cache key for this segment
            segment_cache_key = self._generate_cache_key(
                text=segment_text,
                speaker_audio=speaker_audio,
                emotion_audio=emotion_audio,
                **kwargs
            )

            # Check cache first
            cached_segment_audio = self.audio_cache.get_cached_audio(segment_cache_key)
            if cached_segment_audio:
                print(f"💾 Using cached IndexTTS-2 segment for '{character_name}': '{segment_text[:30]}...'")
                segment_audio = cached_segment_audio[0]
            else:
                # Generate audio for this segment with OOM protection
                try:
                    segment_audio = self.engine.generate(
                        text=segment_text,
                        speaker_audio=speaker_audio,
                        emotion_audio=emotion_audio,
                        **kwargs
                    )
                except torch.OutOfMemoryError as e:
                    # Analyze audio after OOM in multi-character segments
                    audio_analysis = self._analyze_audio_after_oom(speaker_audio, emotion_audio, kwargs.get('max_mel_tokens', 1500))
                    raise RuntimeError(
                        f"❌ IndexTTS-2 OOM error in character segment '{character_name}'.\n{audio_analysis}"
                    ) from e

                # Cache the segment result
                duration = segment_audio.shape[-1] / 22050.0  # IndexTTS-2 uses 22050 Hz
                self.audio_cache.cache_audio(segment_cache_key, segment_audio, duration)
                # print(f"💾 IndexTTS-2 CACHED segment for '{character_name}' with key: {segment_cache_key}")
            
            audio_segments.append(segment_audio)
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = torch.cat(audio_segments, dim=-1)
            return combined_audio
        else:
            # Return silence if no segments generated
            return torch.zeros(1, 22050, dtype=torch.float32)
    
    def _generate_cache_key(self, **params) -> str:
        """Generate cache key for IndexTTS-2."""
        return self.audio_cache.generate_cache_key('index_tts', **params)

    def _analyze_audio_after_oom(self, speaker_audio: str, emotion_audio: str, max_mel_tokens: int) -> str:
        """
        Analyze audio files after OOM to provide helpful feedback.
        Uses fast header-based duration check to avoid loading large files.

        Args:
            speaker_audio: Path to speaker audio file
            emotion_audio: Path to emotion audio file
            max_mel_tokens: Current max_mel_tokens setting

        Returns:
            Formatted analysis message with recommendations
        """
        analysis_parts = []

        # Analyze speaker audio
        if speaker_audio and os.path.exists(speaker_audio):
            try:
                import soundfile as sf
                info = sf.info(speaker_audio)
                duration = info.duration

                if duration > 60:
                    analysis_parts.append(f"🔍 Speaker audio: {duration:.1f}s (TOO LONG - major OOM risk)")
                    analysis_parts.append("   ✅ Solution: Use audio under 30 seconds")
                elif duration > 30:
                    analysis_parts.append(f"🔍 Speaker audio: {duration:.1f}s (long - may cause OOM)")
                    analysis_parts.append("   ✅ Recommendation: Use shorter audio for stability")
                else:
                    analysis_parts.append(f"🔍 Speaker audio: {duration:.1f}s (length OK)")

            except Exception:
                analysis_parts.append("🔍 Speaker audio: Could not analyze duration")

        # Analyze emotion audio
        if emotion_audio and os.path.exists(emotion_audio):
            try:
                import soundfile as sf
                info = sf.info(emotion_audio)
                duration = info.duration

                if duration > 60:
                    analysis_parts.append(f"🔍 Emotion audio: {duration:.1f}s (TOO LONG - major OOM risk)")
                elif duration > 30:
                    analysis_parts.append(f"🔍 Emotion audio: {duration:.1f}s (long - may cause OOM)")
                else:
                    analysis_parts.append(f"🔍 Emotion audio: {duration:.1f}s (length OK)")

            except Exception:
                analysis_parts.append("🔍 Emotion audio: Could not analyze duration")

        # Analyze max_mel_tokens setting
        if max_mel_tokens > 1500:
            analysis_parts.append(f"🔍 max_mel_tokens: {max_mel_tokens} (high - may cause OOM)")
            analysis_parts.append("   ✅ Try reducing to 1000-1500")
        else:
            analysis_parts.append(f"🔍 max_mel_tokens: {max_mel_tokens} (setting OK)")

        # Add general recommendations
        analysis_parts.append("")
        analysis_parts.append("💡 Quick fixes:")
        analysis_parts.append("   • Use audio clips under 30 seconds")
        analysis_parts.append("   • Reduce max_mel_tokens to 1000")
        analysis_parts.append("   • Close other GPU applications")

        return "\n".join(analysis_parts)

    def _get_stable_audio_identifier(self, audio_path: str) -> str:
        """
        Get stable identifier for audio file using centralized audio hashing.
        """
        if not audio_path:
            return audio_path

        # Use our centralized audio hashing utility
        from utils.audio.audio_hash import generate_stable_audio_component
        return generate_stable_audio_component(audio_file_path=audio_path)
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return ["wav", "mp3", "flac", "ogg"]
    
    def get_sample_rate(self) -> int:
        """Get native sample rate."""
        return 22050
    
    def get_emotion_labels(self) -> List[str]:
        """Get supported emotion labels."""
        return ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
    
    def create_emotion_vector(self, **emotions) -> List[float]:
        """
        Create emotion vector from keyword arguments.
        
        Args:
            **emotions: Emotion intensities (e.g., happy=0.8, angry=0.2)
            
        Returns:
            List of 8 emotion values
        """
        if self.engine:
            return self.engine.create_emotion_vector(**emotions)
        else:
            # Fallback implementation
            labels = self.get_emotion_labels()
            vector = [0.0] * len(labels)
            for i, label in enumerate(labels):
                if label in emotions:
                    vector[i] = max(0.0, min(1.2, float(emotions[label])))
            return vector
    
    def unload(self):
        """Unload the engine to free memory."""
        if self.engine:
            self.engine.unload()
            self.engine = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.unload()