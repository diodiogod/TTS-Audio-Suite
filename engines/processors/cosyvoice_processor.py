"""
CosyVoice3 Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using CosyVoice3 engine

Provides:
- Character switching with [CharacterName] syntax
- Pause tag support ([pause:2s], [wait:500ms])
- Chunking for long text
- Caching integration
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.text.character_parser import CharacterParser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.voice.discovery import get_character_mapping, get_available_characters, voice_discovery
from engines.adapters.cosyvoice_adapter import CosyVoiceAdapter


class CosyVoiceProcessor:
    """
    Internal processor for CosyVoice3 TTS generation.
    Handles character processing, pause tags, and generation orchestration.
    """

    SAMPLE_RATE = 24000  # CosyVoice3 native sample rate

    def __init__(self, engine_config: Dict[str, Any]):
        """
        Initialize CosyVoice3 processor.
        
        Args:
            engine_config: Engine configuration from CosyVoice3 Engine node
        """
        self.engine_config = engine_config
        
        # Extract configuration
        self.model_path = engine_config.get('model_path', 'Fun-CosyVoice3-0.5B')
        self.device = engine_config.get('device', 'auto')
        self.mode = engine_config.get('mode', 'zero_shot')
        self.speed = engine_config.get('speed', 1.0)
        self.use_fp16 = engine_config.get('use_fp16', True)
        self.load_trt = engine_config.get('load_trt', False)
        self.load_vllm = engine_config.get('load_vllm', False)
        self.instruct_text = engine_config.get('instruct_text', '')
        self.reference_text = engine_config.get('reference_text', '')
        
        # Initialize adapter
        self.adapter = CosyVoiceAdapter()
        self.adapter.initialize_engine(
            model_path=self.model_path,
            device=self.device,
            use_fp16=self.use_fp16,
            load_trt=self.load_trt,
            load_vllm=self.load_vllm
        )
        
        # Setup character parser
        self._setup_character_parser()

    def update_config(self, engine_config: Dict[str, Any]):
        """
        Update processor configuration with new parameters.
        Called when cached processor is reused with different settings.
        """
        self.engine_config = engine_config

        # Update extracted configuration
        self.speed = engine_config.get('speed', 1.0)
        self.instruct_text = engine_config.get('instruct_text', '')
        self.reference_text = engine_config.get('reference_text', '')

        # Note: model_path, device, fp16, trt, vllm are not updated
        # as they would require reloading the model

    def _setup_character_parser(self):
        """Set up character parser with available characters and aliases."""
        self.character_parser = CharacterParser()

        # Get available characters and aliases
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        # Build complete available set (like Step Audio EditX does)
        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        # Add "narrator"
        all_available.add("narrator")

        self.character_parser.set_available_characters(list(all_available))

        # Set language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            self.character_parser.set_character_language_default(char, lang)

    def process_text(
        self,
        text: str,
        speaker_audio: Optional[Dict[str, Any]] = None,
        seed: int = 1,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        silence_between_chunks_ms: int = 100,
        enable_pause_tags: bool = True,
        return_info: bool = False,
        character_mapping: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Process text and generate audio with CosyVoice3.
        
        Args:
            text: Input text with potential character tags and pause tags
            speaker_audio: Speaker reference audio tensor dict
            seed: Random seed for reproducibility
            enable_chunking: Enable text chunking for long text
            max_chars_per_chunk: Maximum characters per chunk
            silence_between_chunks_ms: Silence between chunks in ms
            enable_pause_tags: Enable pause tag processing
            return_info: Whether to return generation info
            
        Returns:
            Tuple of (audio_tensor, generation_info or None)
        """
        generation_info = {
            'engine': 'cosyvoice3',
            'mode': self.mode,
            'speed': self.speed,
            'seed': seed,
            'character_segments': [],
            'pause_tags_processed': False,
            'chunks_processed': 0
        }

        # Get speaker audio path
        speaker_audio_path = None
        if speaker_audio:
            if isinstance(speaker_audio, dict):
                speaker_audio_path = speaker_audio.get('audio_path') or speaker_audio.get('waveform_path')
            elif isinstance(speaker_audio, str):
                speaker_audio_path = speaker_audio

        # Parse character segments with parse_text_segments (like Step Audio EditX)
        # Rebuild available characters set each time (like Step Audio EditX does)
        import re

        # Get available characters and aliases
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        # Build complete available set
        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        # Also add characters from text (extract from tags)
        character_tags = re.findall(r'\[([^\]]+)\]', text)
        for tag in character_tags:
            if not tag.startswith('pause:'):
                character_name = tag.split('|')[0].strip().lower()
                all_available.add(character_name)

        # Add "narrator"
        all_available.add("narrator")

        self.character_parser.set_available_characters(list(all_available))

        # Set language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            self.character_parser.set_character_language_default(char, lang)

        self.character_parser.reset_session_cache()

        # Parse segments
        segment_objects = self.character_parser.parse_text_segments(text)

        if len(segment_objects) > 1:
            print(f"🎭 CosyVoice3: Processing {len(segment_objects)} character segment(s)")
            generation_info['character_segments'] = [
                {'character': seg.character, 'text_preview': seg.text[:50]} for seg in segment_objects
            ]

        # Get character mapping for voice switching
        # Use provided character_mapping if available (e.g., from SRT processor with connected narrator)
        # Otherwise auto-discover from character folders
        if character_mapping is None:
            # Use audio_only mode - text is optional and comes from character files if available
            # This works for all 3 CosyVoice modes (zero_shot, instruct2, cross_lingual)
            # IMPORTANT: Use lowercase character names for mapping lookup (character parser returns lowercase)
            unique_characters = set(seg.character.lower() if seg.character else "narrator" for seg in segment_objects)
            unique_characters.add("narrator")
            character_mapping = {}
            if unique_characters:
                character_mapping = get_character_mapping(list(unique_characters), engine_type="audio_only")
                print(f"🎭 Character mapping: {list(character_mapping.keys())}")
                for char, (audio, text) in character_mapping.items():
                    if audio:
                        print(f"   {char}: {audio[:50]}... | Ref: {text[:30] if text else 'None'}...")
        else:
            # Using provided character mapping (e.g., from SRT with connected narrator)
            print(f"🎭 Using provided character mapping: {list(character_mapping.keys())}")
            for char, (audio, text) in character_mapping.items():
                if audio:
                    print(f"   {char}: {audio[:50]}... | Ref: {text[:30] if text else 'None'}...")

        # Check if any segments have shorter text than their reference (warn once)
        short_text_warning_shown = False
        for char, (audio, ref_text) in character_mapping.items():
            if ref_text:
                for seg in segment_objects:
                    seg_char = seg.character.lower() if seg.character else "narrator"
                    if seg_char == char and seg.text and len(seg.text.strip()) < 0.5 * len(ref_text):
                        if not short_text_warning_shown:
                            print(f"⚠️  One or more segments have shorter text than their reference transcripts.")
                            print(f"   This works well but quality might be slightly reduced for very short segments.")
                            short_text_warning_shown = True
                        break
                if short_text_warning_shown:
                    break

        audio_segments = []

        for segment in segment_objects:
            character = segment.character.lower() if segment.character else "narrator"
            segment_text = segment.text
            language = segment.language
            segment_text = segment_text.strip()
            if not segment_text:
                continue

            # Prepend CosyVoice language tag if language is specified
            # Maps segment language to CosyVoice's native <|lang|> format
            language_tag_prefix = ""
            if language:
                from utils.text.cosyvoice_special_tags import COSYVOICE_LANGUAGES
                from utils.models.language_mapper import resolve_language_alias
                import re

                # Check if text already has a VALID CosyVoice language tag (user manually added it)
                # Only skip prepending if it's a supported language: <|en|>, <|zh|>, <|ja|>, <|ko|>
                manual_tag_match = re.match(r'^\s*<\|([a-z]{2})\|>', segment_text)
                has_manual_lang_tag = manual_tag_match and manual_tag_match.group(1) in COSYVOICE_LANGUAGES

                if not has_manual_lang_tag:
                    # Resolve language alias to canonical code (e.g., "german" → "de")
                    canonical_lang = resolve_language_alias(language)

                    # Only prepend non-English language tags (en is CosyVoice default, no tag needed)
                    # This prevents forcing English mode on all text and allows auto-detection
                    if canonical_lang in COSYVOICE_LANGUAGES and canonical_lang != 'en':
                        language_tag_prefix = COSYVOICE_LANGUAGES[canonical_lang] + " "

                    # Prepend language tag to segment text
                    if language_tag_prefix:
                        segment_text = language_tag_prefix + segment_text

            # Apply per-segment parameter overrides (e.g., [seed:42], [speed:1.5])
            segment_seed = seed
            segment_speed = self.speed
            if segment.parameters:
                print(f"  📊 Applying parameters: {segment.parameters}")
                segment_params_applied = apply_segment_parameters(
                    {'seed': seed, 'speed': self.speed},
                    segment.parameters,
                    'cosyvoice'
                )
                segment_seed = segment_params_applied.get('seed', seed)
                segment_speed = segment_params_applied.get('speed', self.speed)

            # Determine speaker audio for this segment
            # Priority: connected opt_narrator > character-specific voices > character mapping narrator fallback
            current_speaker_audio = speaker_audio_path
            current_reference_text = self.reference_text

            # For non-narrator characters: always use character mapping
            # For narrator: use connected audio if available, otherwise use character mapping
            if character in character_mapping:
                # Skip character mapping ONLY if this is narrator AND we have connected speaker audio
                if character == "narrator" and speaker_audio_path:
                    # Use connected narrator voice (already set in current_speaker_audio)
                    pass
                else:
                    # Use character-specific voice from mapping
                    char_audio, char_text = character_mapping[character]
                    if char_audio:
                        current_speaker_audio = char_audio
                        if char_text:
                            current_reference_text = char_text
                        print(f"🎭 Using character voice '{character}'")

            # Process pause tags if enabled
            if enable_pause_tags and PauseTagProcessor.has_pause_tags(segment_text):
                generation_info['pause_tags_processed'] = True
                
                # Define TTS generation function for pause processor
                def tts_generate_func(text_content: str, segment_params: Optional[Dict] = None):
                    return self._generate_audio_segment(
                        text=text_content,
                        speaker_audio=current_speaker_audio,
                        reference_text=current_reference_text,
                        seed=segment_seed,
                        speed=segment_speed
                    )
                
                pause_segments, clean_text = PauseTagProcessor.parse_pause_tags(segment_text)
                segment_audio = PauseTagProcessor.generate_audio_with_pauses(
                    segments=pause_segments,
                    tts_generate_func=tts_generate_func,
                    sample_rate=self.SAMPLE_RATE
                )
            else:
                # Chunking for long text
                if enable_chunking and len(segment_text) > max_chars_per_chunk:
                    chunks = ImprovedChatterBoxChunker.split_into_chunks(
                        segment_text,
                        max_chars=max_chars_per_chunk
                    )
                    generation_info['chunks_processed'] += len(chunks)
                    
                    chunk_audios = []
                    for chunk in chunks:
                        chunk_audio = self._generate_audio_segment(
                            text=chunk,
                            speaker_audio=current_speaker_audio,
                            reference_text=current_reference_text,
                            seed=segment_seed,
                            speed=segment_speed
                        )
                        chunk_audios.append(chunk_audio)
                    
                    # Combine chunks with silence
                    segment_audio = self._combine_with_silence(
                        chunk_audios, 
                        silence_ms=silence_between_chunks_ms
                    )
                else:
                    segment_audio = self._generate_audio_segment(
                        text=segment_text,
                        speaker_audio=current_speaker_audio,
                        reference_text=current_reference_text,
                        seed=segment_seed,
                        speed=segment_speed
                    )

            audio_segments.append(segment_audio)

        # Combine all character segments
        if audio_segments:
            final_audio = self._combine_with_silence(
                audio_segments,
                silence_ms=silence_between_chunks_ms
            )
        else:
            final_audio = torch.zeros(1, self.SAMPLE_RATE, dtype=torch.float32)

        # Normalize dimensions
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0)
        elif final_audio.dim() > 2:
            final_audio = final_audio.squeeze()
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0)

        if return_info:
            return final_audio, generation_info
        return final_audio, None

    def _generate_audio_segment(
        self,
        text: str,
        speaker_audio: Optional[str],
        reference_text: Optional[str],
        seed: int,
        speed: Optional[float] = None
    ) -> torch.Tensor:
        """Generate audio for a single text segment."""
        # Use provided speed or fallback to instance speed
        effective_speed = speed if speed is not None else self.speed

        # Don't pass mode - let adapter auto-detect based on instruct_text/reference_text
        return self.adapter.generate(
            text=text,
            speaker_audio=speaker_audio,
            reference_text=reference_text,
            instruct_text=self.instruct_text,
            speed=effective_speed,
            seed=seed,
            use_fp16=self.use_fp16  # Include fp16 in cache key
        )

    def _combine_with_silence(
        self, 
        audio_segments: List[torch.Tensor],
        silence_ms: int = 100
    ) -> torch.Tensor:
        """Combine audio segments with silence between them."""
        if not audio_segments:
            return torch.zeros(1, 0, dtype=torch.float32)
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Create silence
        silence_samples = int(silence_ms * self.SAMPLE_RATE / 1000)
        
        combined_parts = []
        for i, segment in enumerate(audio_segments):
            # Ensure 2D shape
            if segment.dim() == 1:
                segment = segment.unsqueeze(0)
            
            combined_parts.append(segment)
            
            # Add silence between segments (not after last)
            if i < len(audio_segments) - 1:
                silence = torch.zeros(
                    segment.shape[0], 
                    silence_samples, 
                    device=segment.device, 
                    dtype=segment.dtype
                )
                combined_parts.append(silence)
        
        return torch.cat(combined_parts, dim=-1)

    def combine_audio_segments(
        self,
        segments: List[torch.Tensor],
        method: str = "auto",
        silence_ms: int = 100,
        text_chunks: Optional[List[str]] = None,
        text_length: int = 0,
        return_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Combine audio segments using unified ChunkCombiner.
        
        Args:
            segments: List of audio tensors
            method: Combination method
            silence_ms: Silence between segments
            text_chunks: Text for each segment
            text_length: Total text length
            return_info: Whether to return combination info
            
        Returns:
            Tuple of (combined_audio, combination_info or None)
        """
        from utils.audio.chunk_combiner import ChunkCombiner
        
        # Prepare segment info for combiner
        segment_info = []
        for i, seg in enumerate(segments):
            if seg.dim() == 1:
                seg = seg.unsqueeze(0)
            text = text_chunks[i] if text_chunks and i < len(text_chunks) else ""
            segment_info.append({
                'audio': seg,
                'text': text,
                'index': i
            })
        
        combined, info = ChunkCombiner.combine_chunks(
            segment_info,
            sample_rate=self.SAMPLE_RATE,
            method=method,
            silence_ms=silence_ms,
            return_info=True
        )
        
        if return_info:
            return combined, info
        return combined, None

    def cleanup(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.unload()
            self.adapter = None
