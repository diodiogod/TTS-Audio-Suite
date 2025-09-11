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
from utils.text.character_parser import CharacterParser
from utils.voice.discovery import get_character_mapping
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
        self.character_parser = CharacterParser()
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
        # Auto-download model if not provided
        if model_path is None:
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
        
        print(f"✅ IndexTTS-2 adapter initialized with model at: {model_path}")
    
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
            
        # Parse character switching tags with emotion support
        processed_segments = self._process_character_tags_with_emotions(text)
        
        # For now, use first segment for single-generation compatibility
        # TODO: Implement multi-segment generation for full character switching
        if processed_segments:
            first_segment = processed_segments[0]
            processed_text = ' '.join(seg['text'] for seg in processed_segments if seg['text'].strip())
            character_name = first_segment.get('character')
            emotion_ref = first_segment.get('emotion')
        else:
            processed_text = text
            character_name = None
            emotion_ref = None
        
        # Determine final speaker and emotion audio
        final_speaker_audio = speaker_audio
        final_emotion_audio = emotion_audio
        
        character_mapping = get_character_mapping()
        
        if character_name and character_name in character_mapping:
            final_speaker_audio = character_mapping[character_name]
            print(f"🎭 Using character voice: {character_name} -> {final_speaker_audio}")
                    
        if emotion_ref and emotion_ref in character_mapping:
            final_emotion_audio = character_mapping[emotion_ref]
            print(f"😊 Using emotion reference: {emotion_ref} -> {final_emotion_audio}")
                    
        # Check cache before generation
        cache_key = self._generate_cache_key(
            text=processed_text,
            speaker_audio=final_speaker_audio,
            emotion_audio=final_emotion_audio,
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
            interval_silence=interval_silence
        )
        
        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            character_desc = character_name or 'narrator'
            print(f"💾 Using cached IndexTTS-2 audio for '{character_desc}': '{processed_text[:30]}...'")
            return cached_audio[0]
            
        # Generate audio
        audio = self.engine.generate(
            text=processed_text,
            speaker_audio=final_speaker_audio,
            emotion_audio=final_emotion_audio,
            emotion_alpha=emotion_alpha,
            emotion_vector=emotion_vector,
            use_emotion_text=use_emotion_text,
            emotion_text=emotion_text,
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
            **kwargs
        )
        
        # Cache the result
        duration = audio.shape[-1] / 22050.0  # IndexTTS-2 uses 22050 Hz
        self.audio_cache.cache_audio(cache_key, audio, duration)
        
        return audio
    
    def _process_character_tags_with_emotions(self, text: str) -> List[Dict[str, Any]]:
        """
        Process character switching tags with emotion support using the new modular parser.
        
        Supports flexible syntax:
        - [Alice] → character="Alice"
        - [de:Alice] → language="de", character="Alice"  
        - [Alice:angry_bob] → character="Alice", emotion="angry_bob"
        - [de:Alice:angry_bob] → language="de", character="Alice", emotion="angry_bob"
        
        Args:
            text: Input text with flexible character/emotion tags
            
        Returns:
            List of segment dictionaries with character, text, language, emotion info
        """
        # Use the new emotion-aware parsing method
        segments = self.character_parser.split_by_character_with_emotions(text)
        
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
    
    def _generate_cache_key(self, **params) -> str:
        """Generate cache key for IndexTTS-2."""
        return self.audio_cache.generate_cache_key('index_tts', **params)
    
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