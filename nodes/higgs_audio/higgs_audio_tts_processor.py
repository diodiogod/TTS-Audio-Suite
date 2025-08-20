"""
Higgs Audio TTS Processor - Handles TTS text processing for Higgs Audio 2 TTS engine
Internal processor used by UnifiedTTSTextNode - not a ComfyUI node itself
"""

import torch
import os
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path for imports
import sys
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class HiggsAudioTTSProcessor:
    """
    Higgs Audio TTS Processor - Internal TTS processing engine for Higgs Audio 2
    Handles text processing, character switching, pause tags, and audio generation
    """
    
    def __init__(self, engine_wrapper):
        """
        Initialize the TTS processor
        
        Args:
            engine_wrapper: HiggsAudioWrapper instance with adapter and config
        """
        self.engine_wrapper = engine_wrapper
        self.sample_rate = 24000
    
    def generate_tts_speech(self, text: str, multi_speaker_mode: str,
                           audio_tensor: Optional[Dict] = None, reference_text: str = "",
                           seed: int = 1, enable_audio_cache: bool = True,
                           max_chars_per_chunk: int = 400, silence_between_chunks_ms: int = 100,
                           **params):
        """
        Process text with Higgs Audio 2 TTS engine
        
        Args:
            text: Text to convert to speech
            multi_speaker_mode: Multi-speaker processing mode
            audio_tensor: Reference audio tensor
            reference_text: Reference text for voice cloning
            seed: Random seed for generation
            enable_audio_cache: Enable audio caching
            max_chars_per_chunk: Maximum characters per chunk (used in adapter)
            silence_between_chunks_ms: Silence between chunks in milliseconds
            **params: Additional parameters
            
        Returns:
            Tuple of (formatted_audio, generation_info)
        """
        try:
            print(f"🎭 Higgs Audio: Using mode '{multi_speaker_mode}'")
            
            if multi_speaker_mode == "Custom Character Switching":
                # Use existing modular utilities - pause processing first, then character parsing (like ChatterBox)
                print(f"🎭 Higgs Audio: Using character switching with pause support")
                
                # Import modular utilities  
                from utils.text.pause_processor import PauseTagProcessor
                from utils.voice.discovery import get_character_mapping
                from utils.text.character_parser import parse_character_text
                
                # Discover characters and build voice mapping
                character_segments = parse_character_text(text)
                all_characters = set(char for char, _ in character_segments)
                character_mapping = get_character_mapping(list(all_characters), engine_type="higgs_audio")
                
                print(f"🎭 Higgs Audio: Processing {len(character_segments)} character segment(s) - {', '.join(sorted(all_characters))}")
                
                # Build voice references - CRITICAL: Start with narrator using connected voice
                narrator_voice_dict = None
                if audio_tensor is not None:
                    narrator_voice_dict = {"waveform": audio_tensor["waveform"], "sample_rate": audio_tensor["sample_rate"]}
                
                voice_refs = {'narrator': narrator_voice_dict}
                
                for character in all_characters:
                    # Skip narrator - already set above with connected voice
                    if character.lower() == "narrator":
                        continue
                        
                    audio_path, _ = character_mapping.get(character, (None, None))
                    if audio_path and os.path.exists(audio_path):
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(audio_path)
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        voice_refs[character] = {"waveform": waveform, "sample_rate": sample_rate}
                    else:
                        # Use main narrator voice as fallback
                        voice_refs[character] = narrator_voice_dict
                
                def tts_generate_func(text_content: str) -> torch.Tensor:
                    """TTS generation function for pause tag processor"""
                    if '[' in text_content and ']' in text_content:
                        # Handle character switching within this segment
                        char_segments = parse_character_text(text_content)
                        segment_audio_parts = []
                        
                        for character, segment_text in char_segments:
                            char_audio_dict = voice_refs.get(character)
                            char_ref_text = reference_text or ""
                            
                            segment_result = self.engine_wrapper.generate_tts_audio(
                                text=segment_text,
                                char_audio=char_audio_dict,
                                char_text=char_ref_text,
                                character=character,
                                seed=seed,
                                enable_audio_cache=enable_audio_cache,
                                max_chars_per_chunk=max_chars_per_chunk,
                                silence_between_chunks_ms=0
                            )
                            segment_audio_parts.append(segment_result)
                        
                        # Combine character segments
                        if segment_audio_parts:
                            return torch.cat(segment_audio_parts, dim=-1)
                        else:
                            return torch.zeros(1, 0)
                    else:
                        # Simple text segment without character switching - use narrator voice
                        narrator_audio = voice_refs.get("narrator")
                        if narrator_audio is None and audio_tensor is not None:
                            narrator_audio = {"waveform": audio_tensor, "sample_rate": 24000}
                        
                        return self.engine_wrapper.generate_tts_audio(
                            text=text_content,
                            char_audio=narrator_audio,
                            char_text=reference_text or "",
                            character="narrator",
                            seed=seed,
                            enable_audio_cache=enable_audio_cache,
                            max_chars_per_chunk=max_chars_per_chunk,
                            silence_between_chunks_ms=0
                        )
                
                # Process with pause tag handling using existing utility
                pause_processor = PauseTagProcessor()
                
                # Parse text into segments (text and pause segments)
                segments, clean_text = pause_processor.parse_pause_tags(text)
                
                # Generate audio with pauses
                if segments:
                    result = pause_processor.generate_audio_with_pauses(
                        segments=segments,
                        tts_generate_func=tts_generate_func,
                        sample_rate=self.sample_rate
                    )
                else:
                    # No pause tags, just generate directly
                    result = tts_generate_func(text)
            
            else:
                # Native multi-speaker modes - process entire conversation as single unit
                print(f"🎭 Higgs Audio: Using native multi-speaker mode (whole conversation processing)")
                
                # Get second narrator audio if provided
                opt_second_narrator = self.engine_wrapper.config.get("opt_second_narrator")
                
                # Prepare reference audios for native mode
                reference_audio_dict = None
                second_audio_dict = None
                
                if audio_tensor is not None:
                    reference_audio_dict = {
                        "waveform": audio_tensor,
                        "sample_rate": 24000
                    }
                
                if opt_second_narrator is not None:
                    second_audio_dict = opt_second_narrator
                
                # Process entire conversation as single unit - let Higgs Audio handle pauses and speaker transitions
                print(f"🎭 Processing full conversation: '{text[:100]}...'")
                
                result = self.engine_wrapper.generate_tts_audio(
                    text=text,  # Full conversation text
                    char_audio=reference_audio_dict,
                    char_text="",  # Higgs Audio doesn't need reference text
                    character="SPEAKER0",
                    seed=seed,
                    enable_audio_cache=enable_audio_cache,
                    max_chars_per_chunk=max_chars_per_chunk,  # This might trigger chunking - unknown how it will interact
                    silence_between_chunks_ms=0,
                    # Native mode specific parameters
                    multi_speaker_mode=multi_speaker_mode,
                    second_narrator_audio=second_audio_dict,
                    second_narrator_text=""  # Higgs Audio doesn't need reference text
                )
            
            # CRITICAL FIX: Ensure tensor has correct dimensions for ComfyUI
            if isinstance(result, torch.Tensor):
                print(f"🔧 Higgs Audio tensor before fix: {result.shape}")
                if result.dim() == 3 and result.size(0) == 1:
                    result = result.squeeze(0)  # Remove batch dimension [1,1,N] -> [1,N]
                    print(f"🔧 Higgs Audio tensor after fix: {result.shape}")
                elif result.dim() == 1:
                    result = result.unsqueeze(0)  # Add channel dimension [N] -> [1,N]
            
            # Convert single tensor result to ComfyUI audio format
            if isinstance(result, torch.Tensor):
                # Ensure correct dimensions for ComfyUI: [channels, samples]
                if result.dim() == 1:
                    result = result.unsqueeze(0)  # Add channel dimension: [samples] -> [1, samples]
                elif result.dim() == 3:
                    # Handle [batch, channels, samples] -> [channels, samples]
                    if result.size(0) == 1:  # batch size of 1
                        result = result.squeeze(0)  # Remove batch dimension
                    else:
                        result = result[0]  # Take first item from batch
                
                # Ensure we have exactly 2 dimensions: [channels, samples]
                if result.dim() != 2:
                    print(f"⚠️ Unexpected tensor dimensions: {result.shape}, reshaping to 2D")
                    if result.dim() == 1:
                        result = result.unsqueeze(0)
                    elif result.dim() > 2:
                        # Flatten to 1D and add channel dimension
                        result = result.view(-1).unsqueeze(0)
                
                # Format using audio processing utilities
                from utils.audio.processing import AudioProcessingUtils
                formatted_audio = AudioProcessingUtils.format_for_comfyui(result.cpu(), self.sample_rate)
                
                print(f"🔧 Higgs Audio raw tensor: {result.shape}")
                print(f"🔧 After format_for_comfyui: {formatted_audio['waveform'].shape}")
                
                generation_info = "Higgs Audio generation completed"
                return (formatted_audio, generation_info)
            
            # Handle non-tensor results (shouldn't happen but safety check)
            else:
                print(f"⚠️ Unexpected result type: {type(result)}")
                return result
                
        except Exception as e:
            print(f"❌ Higgs Audio TTS processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty audio and error info
            from utils.audio.processing import AudioProcessingUtils
            empty_audio = AudioProcessingUtils.create_silence(1.0, self.sample_rate)
            empty_comfy = AudioProcessingUtils.format_for_comfyui(empty_audio, self.sample_rate)
            
            return (empty_comfy, f"Higgs Audio TTS error: {e}")