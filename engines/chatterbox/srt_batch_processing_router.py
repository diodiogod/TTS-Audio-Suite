"""
SRT Batch Processing Router for ChatterBox
Routes SRT processing between streaming (batch) and traditional (sequential) modes
"""

from typing import List, Tuple, Dict, Any
import torch


class SRTBatchProcessingRouter:
    """
    Routes SRT subtitle processing between streaming (batch) and traditional (sequential) modes.
    Reuses existing streaming infrastructure from chatterbox_tts_node.py
    """
    
    def __init__(self, srt_node):
        """Initialize with reference to the SRT node for access to existing methods."""
        self.srt_node = srt_node
    
    def process_subtitles(self, subtitles, subtitle_language_groups, batch_size, **kwargs):
        """
        Process SRT subtitles using streaming or traditional approach.
        
        Args:
            subtitles: Parsed SRT subtitles
            subtitle_language_groups: Language-grouped subtitles  
            batch_size: Batch size for processing decision
            **kwargs: All other processing parameters
            
        Returns:
            (audio_segments, natural_durations, any_segment_cached)
        """
        total_segments = len(subtitles)
        use_streaming = (batch_size > 1)  # Let user decide: batch_size > 1 = streaming, regardless of segment count
        
        if use_streaming:
            print(f"🌊 ChatterBox SRT: Using streaming parallel processing with {batch_size} workers")
            # Pre-load models for streaming efficiency (same as TTS Text node)
            print(f"🚀 SRT STREAMING: Pre-loading models for {len(subtitle_language_groups)} languages")
            self.srt_node._preload_language_models(subtitle_language_groups.keys(), kwargs.get('device', 'auto'))
            return self._process_streaming(subtitles, subtitle_language_groups, batch_size, **kwargs)
        else:
            print(f"📖 ChatterBox SRT: Using traditional sequential processing")  
            return self._process_traditional(subtitles, subtitle_language_groups, **kwargs)
    
    def _process_streaming(self, subtitles, subtitle_language_groups, batch_size, **kwargs):
        """Process using streaming work queue - delegates to existing infrastructure."""
        try:
            from engines.chatterbox.streaming_work_queue import StreamingWorkQueueProcessor
            
            # Convert SRT data to format compatible with existing streaming processor
            language_groups, character_groups_by_lang, voice_refs = self._convert_srt_to_streaming_format(
                subtitles, subtitle_language_groups, kwargs
            )
            
            # Use existing streaming infrastructure 
            inputs = self._build_inputs_dict(kwargs)
            streaming_processor = StreamingWorkQueueProcessor(max_workers=batch_size, tts_node=self.srt_node)
            results = streaming_processor.process_streaming(language_groups, character_groups_by_lang, voice_refs, inputs)
            
            # Convert results back to SRT format
            return self._convert_streaming_results_to_srt_format(results, subtitles, kwargs)
            
        except Exception as e:
            print(f"❌ SRT streaming failed: {e}, falling back to traditional processing")
            return self._process_traditional(subtitles, subtitle_language_groups, **kwargs)
    
    def _process_traditional(self, subtitles, subtitle_language_groups, **kwargs):
        """Process using traditional sequential approach - calls existing SRT node logic."""
        # This delegates to the existing traditional processing logic that's already in the SRT node
        return self.srt_node._process_traditional_srt_logic(
            subtitles=subtitles,
            subtitle_language_groups=subtitle_language_groups,
            language=kwargs['language'],
            device=kwargs['device'],
            exaggeration=kwargs['exaggeration'],
            temperature=kwargs['temperature'],
            cfg_weight=kwargs['cfg_weight'],
            seed=kwargs['seed'],
            reference_audio=kwargs['reference_audio'],
            audio_prompt_path=kwargs['audio_prompt_path'],
            enable_audio_cache=kwargs['enable_audio_cache'],
            crash_protection_template=kwargs['crash_protection_template'],
            stable_audio_prompt_component=kwargs['stable_audio_prompt_component'],
            all_subtitle_segments=kwargs['all_subtitle_segments'],
            audio_prompt=kwargs['audio_prompt']
        )
    
    def _convert_srt_to_streaming_format(self, subtitles, subtitle_language_groups, kwargs):
        """Convert SRT data structures to format needed by streaming processor."""
        language_groups = {}
        character_groups_by_lang = {}
        
        # Build voice references - handle character switching properly
        audio_prompt_path = kwargs.get('audio_prompt_path', '')
        voice_refs = {'narrator': audio_prompt_path or 'none'}
        
        # Import character discovery for proper voice mapping
        try:
            from utils.voice.discovery import get_available_characters, get_character_mapping
            available_chars = get_available_characters()
            
            # Get character mapping for all available characters
            char_mapping = get_character_mapping(list(available_chars), "chatterbox")
            
            # Build voice references for all characters
            for char in available_chars:
                char_audio_path, char_text = char_mapping.get(char, (audio_prompt_path or 'none', None))
                voice_refs[char] = char_audio_path
        except ImportError:
            pass
        
        # Convert subtitle groups to streaming format following TTS Text node pattern
        # Build language_groups first using TUPLES like TTS Text node does
        for lang_code, lang_subtitles in subtitle_language_groups.items():
            language_groups[lang_code] = []
            
            for i, subtitle, subtitle_type, character_segments_with_lang in lang_subtitles:
                if subtitle_type == 'multilingual' or subtitle_type == 'multicharacter':
                    # Handle complex subtitles with character switching
                    for char, text, seg_lang in character_segments_with_lang:
                        # Use same tuple format as TTS Text node: (idx, char, segment_text, lang)
                        segment_tuple = (i, char, text, seg_lang)
                        language_groups[lang_code].append(segment_tuple)
                        
                        # Ensure voice reference exists for this character
                        if char not in voice_refs:
                            voice_refs[char] = audio_prompt_path or 'none'
                else:
                    # Simple subtitle - single narrator
                    # Use same tuple format as TTS Text node: (idx, char, segment_text, lang)
                    segment_tuple = (i, 'narrator', subtitle.text, lang_code)
                    language_groups[lang_code].append(segment_tuple)
        
        # Now group characters within each language using the same pattern as TTS Text node
        from engines.chatterbox.character_grouper import CharacterGrouper
        character_groups_by_lang = {}
        for lang_code, lang_segments in language_groups.items():
            character_groups_by_lang[lang_code] = CharacterGrouper.group_by_character(lang_segments)
        
        return language_groups, character_groups_by_lang, voice_refs
    
    def _build_inputs_dict(self, kwargs):
        """Build inputs dict for streaming processor from SRT kwargs."""
        return {
            'exaggeration': kwargs.get('exaggeration', 0.5),
            'temperature': kwargs.get('temperature', 0.8), 
            'cfg_weight': kwargs.get('cfg_weight', 0.5),
            'seed': kwargs.get('seed', 0),
            'enable_audio_cache': kwargs.get('enable_audio_cache', True),
            'crash_protection_template': kwargs.get('crash_protection_template', 'hmm ,, {seg} hmm ,,'),
            'device': kwargs.get('device', 'auto')
        }
    
    def _convert_streaming_results_to_srt_format(self, results, subtitles, kwargs):
        """Convert streaming results back to SRT format."""
        audio_segments = [None] * len(subtitles)
        natural_durations = [0.0] * len(subtitles) 
        any_segment_cached = False
        
        # Group results by original subtitle index for character segments that need combining
        subtitle_audio_parts = {}
        for original_idx, audio in results.items():
            if audio is not None:
                if original_idx not in subtitle_audio_parts:
                    subtitle_audio_parts[original_idx] = []
                subtitle_audio_parts[original_idx].append(audio)
                if kwargs.get('enable_audio_cache', True):
                    any_segment_cached = True
        
        # Combine character segments back into complete subtitle audio
        for original_idx, audio_parts in subtitle_audio_parts.items():
            if len(audio_parts) == 1:
                # Simple case - single audio segment
                audio_segments[original_idx] = audio_parts[0]
                natural_durations[original_idx] = float(audio_parts[0].shape[-1]) / 22050.0
            else:
                # Complex case - multiple character segments need combining
                import torch
                combined_audio = torch.cat(audio_parts, dim=-1)
                audio_segments[original_idx] = combined_audio
                natural_durations[original_idx] = float(combined_audio.shape[-1]) / 22050.0
        
        # Handle any missing segments
        for i in range(len(subtitles)):
            if audio_segments[i] is None:
                audio_segments[i] = None
                natural_durations[i] = 0.0
        
        return audio_segments, natural_durations, any_segment_cached