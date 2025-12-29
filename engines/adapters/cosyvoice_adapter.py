"""
CosyVoice3 Engine Adapter

Provides standardized interface for CosyVoice3 integration with TTS Audio Suite.
Handles parameter mapping, character switching, and mode selection.
"""

import os
import torch
from typing import Dict, Any, Optional, List

from engines.cosyvoice.cosyvoice import CosyVoiceEngine
from engines.cosyvoice.cosyvoice_downloader import cosyvoice_downloader
from utils.text.character_parser import character_parser
from utils.voice.discovery import get_character_mapping, get_available_characters
from utils.audio.cache import get_audio_cache


class CosyVoiceAdapter:
    """
    Adapter for CosyVoice3 engine providing unified interface compatibility.
    
    Handles:
    - Parameter mapping between unified interface and CosyVoice3
    - Character switching with [CharacterName] syntax
    - Mode selection (zero_shot, instruct, cross_lingual)
    - Caching integration
    - Model management
    """
    
    def __init__(self):
        """Initialize the CosyVoice3 adapter."""
        self.engine = None
        self.audio_cache = get_audio_cache()
    
    def initialize_engine(self,
                         model_path: Optional[str] = None,
                         device: str = "auto",
                         use_fp16: bool = True,
                         load_trt: bool = False,
                         load_vllm: bool = False):
        """
        Initialize CosyVoice3 engine.

        Args:
            model_path: Path to model directory (auto-downloaded if None)
            device: Target device
            use_fp16: Use FP16 for inference
            load_trt: Load TensorRT engine
            load_vllm: Load vLLM engine
        """
        # Auto-download model if not provided
        if model_path is None or model_path == "Fun-CosyVoice3-0.5B":
            if not cosyvoice_downloader.is_model_available():
                print("📥 CosyVoice3 model not found, downloading...")
                model_path = cosyvoice_downloader.download_model()
            else:
                model_path = cosyvoice_downloader.get_model_path()

        # Initialize engine
        self.engine = CosyVoiceEngine(
            model_dir=model_path,
            device=device,
            use_fp16=use_fp16,
            load_trt=load_trt,
            load_vllm=load_vllm
        )
    
    def generate(self,
                text: str,
                speaker_audio: Optional[str] = None,
                reference_text: Optional[str] = None,
                instruct_text: Optional[str] = None,
                speed: float = 1.0,
                stream: bool = False,
                **kwargs) -> torch.Tensor:
        """
        Generate speech with CosyVoice3.

        Mode is auto-detected:
        - If instruct_text provided → instruct mode
        - If reference_text from Character Voices → zero_shot mode
        - Otherwise → cross_lingual mode

        Args:
            text: Text to synthesize (supports [character] tags)
            speaker_audio: Speaker reference audio file path
            reference_text: Transcript of reference audio (from Character Voices .txt)
            instruct_text: Optional instruction for dialect/emotion/speed control
            speed: Speech speed multiplier (0.5-2.0)
            stream: Enable streaming
            **kwargs: Additional parameters

        Returns:
            Generated audio tensor [1, samples] at 24000 Hz
        """
        if self.engine is None:
            self.initialize_engine()

        # Auto-detect mode based on available parameters
        print(f"🔍 Mode detection - instruct_text: '{instruct_text}', reference_text: '{reference_text[:50] if reference_text else None}...'")
        if instruct_text and instruct_text.strip():
            mode = "instruct"
            print(f"✅ Selected mode: instruct")
            if reference_text and reference_text.strip():
                print("⚠️ CosyVoice3: Both instruction and transcript provided. Using instruction mode (transcript ignored for best dialect/emotion control).")
        elif reference_text and reference_text.strip():
            mode = "zero_shot"
            print(f"✅ Selected mode: zero_shot")
        else:
            mode = "cross_lingual"
            print(f"✅ Selected mode: cross_lingual")

        # Check if text contains character tags
        has_character_tags = character_parser.CHARACTER_TAG_PATTERN.search(text) is not None

        if has_character_tags:
            # Parse character switching tags
            processed_segments = self._process_character_tags(text)
            
            if len(processed_segments) > 1:
                # Multi-segment character switching
                return self._generate_multi_character_segments(
                    processed_segments,
                    speaker_audio,
                    reference_text,
                    instruct_text=instruct_text,
                    speed=speed,
                    **kwargs
                )
            elif processed_segments:
                # Single character segment
                first_segment = processed_segments[0]
                processed_text = first_segment.get('text', '').strip()
                character_name = first_segment.get('character')
            else:
                processed_text = text
                character_name = None
        else:
            processed_text = text
            character_name = None

        # Determine final speaker audio
        final_speaker_audio = speaker_audio
        final_reference_text = reference_text

        # Only do character mapping if we have character tags
        if has_character_tags and character_name:
            character_mapping = get_character_mapping([character_name], engine_type="cosyvoice")
            
            if character_name in character_mapping:
                char_audio, char_text = character_mapping[character_name]
                if char_audio is not None:
                    final_speaker_audio = char_audio
                    # Use character's reference text if available
                    if char_text:
                        final_reference_text = char_text
                    print(f"🎭 Using character voice: {character_name} -> {final_speaker_audio}")

        # Generate cache key (mode is deterministic from reference_text + instruct_text)
        cache_key = self._generate_cache_key(
            text=processed_text,
            speaker_audio=self._get_stable_audio_identifier(final_speaker_audio),
            reference_text=final_reference_text,
            instruct_text=instruct_text,
            speed=speed,
            **kwargs
        )

        # Check cache
        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            print(f"💾 Using cached CosyVoice3 audio for: '{processed_text[:30]}...'")
            return cached_audio[0]

        # Handle seed for deterministic generation
        seed = kwargs.get('seed', 0)
        if seed != 0:
            torch.manual_seed(seed)
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

        # Convert CosyVoice paralinguistic tags from <tag> to [tag]
        from utils.text.cosyvoice_special_tags import convert_cosyvoice_special_tags
        processed_text = convert_cosyvoice_special_tags(processed_text)

        # Generate audio
        audio = self.engine.generate(
            text=processed_text,
            prompt_wav=final_speaker_audio,
            prompt_text=final_reference_text,
            mode=mode,
            instruct_text=instruct_text,
            speed=speed,
            stream=stream
        )

        # Cache the result
        duration = audio.shape[-1] / 24000.0  # CosyVoice3 uses 24000 Hz
        self.audio_cache.cache_audio(cache_key, audio, duration)

        return audio
    
    def _process_character_tags(self, text: str) -> List[Dict[str, Any]]:
        """Process character switching tags."""
        from utils.text.character_parser.base_parser import CharacterParser
        from utils.voice.discovery import get_available_characters, voice_discovery

        temp_parser = CharacterParser()

        # Get available characters and aliases
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

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

        # Parse segments with language information
        segments = temp_parser.split_by_character(text, include_language=True)

        processed_segments = []
        for character, segment_text, language in segments:
            segment_info = {
                'character': character,
                'text': segment_text,
                'language': language
            }
            processed_segments.append(segment_info)

        return processed_segments
    
    def _generate_multi_character_segments(
        self, 
        segments: List[Dict[str, Any]], 
        default_speaker_audio: Optional[str], 
        default_reference_text: Optional[str],
        **kwargs
    ) -> torch.Tensor:
        """Generate audio for multiple character segments and combine them."""
        audio_segments = []

        # Get character mapping for all unique characters
        unique_characters = set()
        for segment in segments:
            if segment.get('character'):
                unique_characters.add(segment['character'])

        character_mapping = {}
        if unique_characters:
            character_mapping = get_character_mapping(list(unique_characters), engine_type="cosyvoice")

        print(f"🎭 CosyVoice3: Processing {len(segments)} character segment(s)")

        for segment in segments:
            character_name = segment.get('character', 'narrator')
            segment_text = segment.get('text', '').strip()

            if not segment_text:
                continue

            # Determine speaker audio for this segment
            speaker_audio = default_speaker_audio
            reference_text = default_reference_text

            if character_name and character_name in character_mapping:
                char_audio, char_text = character_mapping[character_name]
                if char_audio:
                    speaker_audio = char_audio
                    if char_text:
                        reference_text = char_text
                    print(f"📖 Using character voice '{character_name}'")

            # Generate cache key for this segment
            segment_cache_key = self._generate_cache_key(
                text=segment_text,
                speaker_audio=self._get_stable_audio_identifier(speaker_audio),
                reference_text=reference_text,
                **kwargs
            )

            # Check cache first
            cached_segment_audio = self.audio_cache.get_cached_audio(segment_cache_key)
            if cached_segment_audio:
                print(f"💾 Using cached CosyVoice3 segment for '{character_name}'")
                segment_audio = cached_segment_audio[0]
            else:
                # Convert CosyVoice paralinguistic tags from <tag> to [tag]
                from utils.text.cosyvoice_special_tags import convert_cosyvoice_special_tags
                segment_text_converted = convert_cosyvoice_special_tags(segment_text)

                # Generate audio for this segment
                segment_audio = self.engine.generate(
                    text=segment_text_converted,
                    prompt_wav=speaker_audio,
                    prompt_text=reference_text,
                    **kwargs
                )

                # Cache the segment
                duration = segment_audio.shape[-1] / 24000.0  # CosyVoice3 uses 24000 Hz
                self.audio_cache.cache_audio(segment_cache_key, segment_audio, duration)

            audio_segments.append(segment_audio)

        # Combine all audio segments
        if audio_segments:
            combined_audio = torch.cat(audio_segments, dim=-1)
            return combined_audio
        else:
            return torch.zeros(1, 24000, dtype=torch.float32)  # 1 second at CosyVoice3's 24000 Hz
    
    def _generate_cache_key(self, **params) -> str:
        """Generate cache key for CosyVoice3."""
        return self.audio_cache.generate_cache_key('cosyvoice', **params)

    def _get_stable_audio_identifier(self, audio_path: str) -> str:
        """Get stable identifier for audio file using centralized audio hashing."""
        if not audio_path:
            return audio_path

        from utils.audio.audio_hash import generate_stable_audio_component
        return generate_stable_audio_component(audio_file_path=audio_path)
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return ["wav", "mp3", "flac", "ogg"]
    
    def get_sample_rate(self) -> int:
        """Get native sample rate."""
        return 24000
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return CosyVoiceEngine.SUPPORTED_LANGUAGES.copy()
    
    def get_supported_dialects(self) -> List[str]:
        """Get supported Chinese dialects."""
        return CosyVoiceEngine.SUPPORTED_DIALECTS.copy()
    
    def unload(self):
        """Unload the engine to free memory."""
        if self.engine:
            self.engine.unload()
            self.engine = None
