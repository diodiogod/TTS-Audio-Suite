"""
Multilingual Engine - Central orchestrator for multilingual TTS generation
Handles language switching, character management, and cache optimization for any TTS engine
"""

import torch
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass

from utils.text.character_parser import character_parser
from utils.voice.discovery import get_available_characters, get_character_mapping
from utils.voice.character_logging import resolved_character_label
from utils.text.pause_processor import PauseTagProcessor


@dataclass
class AudioSegmentResult:
    """Result of audio generation for a single segment"""
    audio: torch.Tensor
    duration: float
    character: str
    text: str
    language: str
    original_index: int
    edit_tags: list = None  # Edit tags for this specific segment


@dataclass
class MultilingualResult:
    """Complete result of multilingual processing"""
    audio: torch.Tensor
    total_duration: float
    segments: List[AudioSegmentResult]
    languages_used: List[str]
    characters_used: List[str]
    info_message: str


class MultilingualEngine:
    """
    Central orchestrator for multilingual TTS generation.
    
    Handles:
    - Language-aware character parsing
    - Smart model loading with cache optimization
    - Language grouping for efficient processing
    - Proper segment ordering and audio assembly
    """
    
    def __init__(self, engine_type: str):
        """
        Initialize multilingual engine.
        
        Args:
            engine_type: "f5tts" or "chatterbox"
        """
        self.engine_type = engine_type
        self.sample_rate = 24000 if engine_type == "f5tts" else 44100

        
    def process_multilingual_text(self, text: str, engine_adapter, **params) -> MultilingualResult:
        """
        Main entry point for multilingual processing.
        
        Args:
            text: Input text with character/language tags
            engine_adapter: Engine-specific adapter (F5TTSEngineAdapter or ChatterBoxEngineAdapter)
            **params: Engine-specific parameters
            
        Returns:
            MultilingualResult with generated audio and metadata
        """
        # 1. Parse character segments with languages from original text
        character_segments_with_lang = character_parser.split_by_character_with_language(text)

        # Get detailed segments to access original character information
        detailed_segments = character_parser.parse_text_segments(text)

        # Extract edit tags per segment and handle pause-splitting if needed
        segment_edit_tags = []
        if self.engine_type == "chatterbox":
            # Check if we need edit tag extraction (only for classic ChatterBox, not official 23-lang)
            extract_edit_for_chatterbox = params.get('extract_edit_tags', False)
            if extract_edit_for_chatterbox:
                from utils.text.step_audio_editx_special_tags import parse_edit_tags_with_iterations
                from dataclasses import replace

                # First pass: extract edit tags and check for pause tags
                new_detailed_segments = []
                for seg_idx, seg in enumerate(detailed_segments):
                    # Extract edit tags from this segment's original text
                    seg_clean_text, seg_edits = parse_edit_tags_with_iterations(seg.text)

                    # Check if this segment has BOTH pause tags and edit tags
                    if seg_edits and PauseTagProcessor.has_pause_tags(seg_clean_text):
                        # Split by pause tags FIRST to preserve pauses during edit processing
                        pause_segments, _ = PauseTagProcessor.parse_pause_tags(seg_clean_text)

                        # Create new segments for each pause-split part
                        for pause_seg_type, pause_content in pause_segments:
                            if pause_seg_type == 'text':
                                # Text segment - create new detailed segment
                                new_seg = replace(seg, text=pause_content)
                                new_detailed_segments.append(new_seg)
                                segment_edit_tags.append(seg_edits)  # This text segment can be edited
                            elif pause_seg_type == 'pause':
                                # Pause segment - create silence marker
                                pause_marker_seg = replace(seg, text=f"__PAUSE_{pause_content}__")
                                new_detailed_segments.append(pause_marker_seg)
                                segment_edit_tags.append([])  # Silence has no edit tags
                    else:
                        # No pause or no edit tags - keep segment as-is
                        seg.text = seg_clean_text
                        new_detailed_segments.append(seg)
                        segment_edit_tags.append(seg_edits)

                # Replace detailed_segments with pause-split version
                detailed_segments = new_detailed_segments
            else:
                segment_edit_tags = [[] for _ in detailed_segments]
        else:
            segment_edit_tags = [[] for _ in detailed_segments]
        
        # 2. Analyze segments
        characters = list(set(char for char, _, _ in character_segments_with_lang))
        languages = list(set(lang for _, _, lang in character_segments_with_lang))
        has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
        has_multiple_languages = len(languages) > 1
        
        # Print analysis
        if has_multiple_languages:
            print(f"🌍 {self.engine_type.title()}: Language switching mode - found languages: {', '.join(languages)}")
        if has_multiple_characters:
            print(f"🎭 {self.engine_type.title()}: Character switching mode - found characters: {', '.join(characters)}")
        
        # 3. Group segments by language to optimize model loading
        language_groups = self._group_segments_by_language_with_original(detailed_segments)
        
        # 4. Get character mapping for all characters
        character_mapping = get_character_mapping(characters, engine_type=self.engine_type)
        
        # 5. Check cache optimization opportunities
        cache_info = self._analyze_cache_coverage(language_groups, character_mapping, engine_adapter, **params)
        
        # 6. Process each language group with smart model loading  
        all_audio_segments = []
        for lang_code, lang_segments in language_groups.items():
            required_model = engine_adapter.get_model_for_language(
                lang_code, params.get("model", "default")
            )
            if cache_info[lang_code]["all_cached"]:
                print(f"💾 Skipping model load for language '{lang_code}' - all speech cached")
            else:
                # Let the adapter/model manager reuse or restore the requested model.
                # A historical set of loaded names is not evidence it is still resident.
                engine_adapter.load_base_model(required_model, params.get("device", "auto"))
                engine_adapter.node.current_language = required_model
                engine_adapter.node.current_model_name = required_model
            
            # Process each segment in this language group
            for segment_data in lang_segments:
                original_idx, character, segment_text, segment_lang, original_character = segment_data
                segment_display_idx = original_idx + 1  # For display (1-based)

                # Check if this is a pause marker segment
                if segment_text.startswith("__PAUSE_") and segment_text.endswith("__"):
                    # Extract pause duration and create silence
                    pause_duration = float(segment_text[8:-2])  # Extract from __PAUSE_X__
                    silence = PauseTagProcessor.create_silence_segment(pause_duration, self.sample_rate)

                    # Get edit tags for this segment (should be empty for pause)
                    seg_edit_tags = segment_edit_tags[original_idx] if original_idx < len(segment_edit_tags) else []

                    # Store silence segment
                    all_audio_segments.append(AudioSegmentResult(
                        audio=silence,
                        duration=pause_duration,
                        character=character,
                        text="",  # Empty text for silence
                        language=segment_lang,
                        original_index=original_idx,
                        edit_tags=seg_edit_tags
                    ))
                    continue  # Skip TTS generation for pause segments
                
                char_audio, char_text, cache_character = self._resolve_segment_voice(
                    character, original_character, character_mapping, params
                )
                
                segment_audio = cache_info[lang_code]["audio"].get(original_idx)
                if segment_audio is None:
                    # Show generation message with character and language info
                    # Check if we're using main voice for narrator (language-only tags)
                    is_using_main_voice = (self.engine_type == "chatterbox" and
                                         original_character == "narrator" and
                                         params.get("main_audio_reference"))

                    if is_using_main_voice:
                        # Language-only tag using main voice
                        if segment_lang != 'en':
                            print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} using main voice in {segment_lang}...")
                        else:
                            print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} using main voice...")
                    elif character == "narrator":
                        if segment_lang != 'en':
                            print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx} in {segment_lang}...")
                        else:
                            print(f"🎤 Generating {self.engine_type.title()} segment {segment_display_idx}...")
                    else:
                        if segment_lang != 'en':
                            print(f"🎭 Generating {self.engine_type.title()} segment {segment_display_idx} using '{resolved_character_label(character, char_audio)}' in {segment_lang}")
                        else:
                            print(f"🎭 Generating {self.engine_type.title()} segment {segment_display_idx} using '{resolved_character_label(character, char_audio)}'")

                    # Show what model is actually being used for generation (for verification)
                    current_model = getattr(engine_adapter.node, 'current_language', 'unknown')
                    print(f"🔧 ACTUAL MODEL: Generating segment {segment_display_idx} using '{current_model}' model")

                    # Show the final text that will go to the TTS model
                    print(f"🔤 Final text to {self.engine_type.upper()} via multilingual engine ({resolved_character_label(character, char_audio)}): '{segment_text}'")
                    updated_params = params.copy()
                    updated_params["current_language"] = getattr(
                        engine_adapter.node, "current_language", required_model
                    )
                    updated_params["enable_pause_tags"] = True
                    if self.engine_type == "f5tts":
                        updated_params["char_text"] = char_text
                    segment_audio = engine_adapter.generate_segment_audio(
                        text=segment_text, char_audio=char_audio,
                        character=cache_character, **updated_params
                    )
                
                # Calculate duration
                duration = self._get_audio_duration(segment_audio)

                # Get edit tags for this segment
                seg_edit_tags = segment_edit_tags[original_idx] if original_idx < len(segment_edit_tags) else []

                # Store result with original index for proper ordering
                all_audio_segments.append(AudioSegmentResult(
                    audio=segment_audio,
                    duration=duration,
                    character=character,
                    text=segment_text,
                    language=segment_lang,
                    original_index=original_idx,
                    edit_tags=seg_edit_tags
                ))
        
        # 7. Reorder segments back to original order and combine
        all_audio_segments.sort(key=lambda x: x.original_index)
        ordered_audio = [seg.audio for seg in all_audio_segments]
        combined_audio = torch.cat(ordered_audio, dim=1) if ordered_audio else torch.zeros(1, 0)
        
        # 8. Calculate total duration and create info message
        total_duration = sum(seg.duration for seg in all_audio_segments)
        info_message = self._generate_info_message(
            total_duration, len(all_audio_segments), characters, languages, 
            has_multiple_characters, has_multiple_languages
        )
        
        return MultilingualResult(
            audio=combined_audio,
            total_duration=total_duration,
            segments=all_audio_segments,
            languages_used=languages,
            characters_used=characters,
            info_message=info_message
        )
    
    def _group_segments_by_language(self, character_segments_with_lang: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[int, str, str, str]]]:
        """Group character segments by language for efficient processing."""
        language_groups = {}
        for original_idx, (character, segment_text, segment_lang) in enumerate(character_segments_with_lang):
            if segment_lang not in language_groups:
                language_groups[segment_lang] = []
            language_groups[segment_lang].append((original_idx, character, segment_text, segment_lang))
        return language_groups
    
    def _group_segments_by_language_with_original(self, segments: List) -> Dict[str, List[Tuple[int, str, str, str, str]]]:
        """Group segments by language for efficient processing, preserving original character info."""
        language_groups = {}
        for original_idx, segment in enumerate(segments):
            segment_lang = segment.language
            if segment_lang not in language_groups:
                language_groups[segment_lang] = []
            language_groups[segment_lang].append((
                original_idx, 
                segment.character, 
                segment.text, 
                segment_lang, 
                segment.original_character or segment.character
            ))
        return language_groups
    
    def _resolve_segment_voice(self, character, original_character, character_mapping, params):
        """Use identical voice selection for probing and generation."""
        char_audio, char_text = character_mapping.get(character, (None, None))
        cache_character = character
        if self.engine_type == "f5tts":
            if not char_audio or not char_text:
                char_audio = params.get("main_audio_reference")
                char_text = params.get("main_text_reference")
        else:
            main_ref = params.get("main_audio_reference")
            if original_character == "narrator":
                cache_character = "narrator"
                if main_ref:
                    char_audio = main_ref
            char_audio = char_audio or main_ref
        return char_audio, char_text, cache_character

    def _analyze_cache_coverage(self, language_groups, character_mapping, engine_adapter, **params):
        """Retain cached tensors so skipping a load never requires the old model."""
        cache_info = {}
        for language, segments in language_groups.items():
            cached_audio = {}
            model = engine_adapter.get_model_for_language(language, params.get("model", "default"))
            for index, character, text, segment_lang, original_character in segments:
                # These markers are assembled locally and never require a model.
                if text.startswith("__PAUSE_") and text.endswith("__"):
                    continue
                audio = self._get_cached_segment(
                    character, original_character, text, model, character_mapping,
                    engine_adapter, **params
                )
                if audio is not None:
                    cached_audio[index] = audio
            speech_count = sum(
                not (segment[2].startswith("__PAUSE_") and segment[2].endswith("__"))
                for segment in segments
            )
            all_cached = len(cached_audio) == speech_count
            cache_info[language] = {
                "all_cached": all_cached, "audio": cached_audio,
                "cached_segments": len(cached_audio), "segments_count": len(segments),
            }
            if all_cached:
                print(f"💾 Cache optimization: All {speech_count} speech segments in '{language}' are cached")
        return cache_info

    def _get_cached_segment(self, character, original_character, text, required_model,
                            character_mapping, engine_adapter, **params):
        if not params.get("enable_audio_cache", True):
            return None
        # Pause-aware generators cache individual text pieces, not this whole string.
        # Keep their existing assembly path rather than treating a partial hit as complete.
        if PauseTagProcessor.has_pause_tags(text):
            return None
        create_cache = getattr(engine_adapter, "create_segment_cache", None)
        if create_cache is None:
            return None
        _, char_text, cache_character = self._resolve_segment_voice(
            character, original_character, character_mapping, params
        )
        cache_params = dict(params, cache_model_name=required_model, cache_probe=True)
        if self.engine_type == "f5tts":
            cache_params["char_text"] = char_text
        cache_fn = create_cache(character=cache_character, **cache_params)
        return cache_fn(text) if cache_fn is not None else None
    
    def _get_audio_duration(self, audio_tensor: torch.Tensor) -> float:
        """Calculate audio duration in seconds."""
        if audio_tensor.dim() == 1:
            num_samples = audio_tensor.shape[0]
        elif audio_tensor.dim() == 2:
            num_samples = audio_tensor.shape[1]  # Assume shape (channels, samples)
        else:
            num_samples = audio_tensor.numel()
        
        return num_samples / self.sample_rate
    
    def _generate_info_message(self, total_duration: float, num_segments: int, 
                             characters: List[str], languages: List[str],
                             has_multiple_characters: bool, has_multiple_languages: bool) -> str:
        """Generate descriptive info message about the generation."""
        character_info = f"characters: {', '.join(characters)}" if has_multiple_characters else "narrator"
        language_info = f" across {len(languages)} languages ({', '.join(languages)})" if has_multiple_languages else ""
        
        return f"Generated {total_duration:.1f}s audio from {num_segments} segments using {character_info}{language_info} ({self.engine_type.title()} models)"
    
    def is_multilingual_or_multicharacter(self, text: str) -> bool:
        """Quick check if text needs multilingual processing."""
        character_segments_with_lang = character_parser.split_by_character_with_language(text)
        characters = list(set(char for char, _, _ in character_segments_with_lang))
        languages = list(set(lang for _, _, lang in character_segments_with_lang))
        
        has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
        has_multiple_languages = len(languages) > 1
        
        return has_multiple_characters or has_multiple_languages
