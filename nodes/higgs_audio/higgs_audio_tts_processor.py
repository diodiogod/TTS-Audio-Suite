"""
Higgs Audio TTS Processor - Handles TTS text processing for Higgs Audio 2 TTS engine
Internal processor used by UnifiedTTSTextNode - not a ComfyUI node itself
"""

import torch
import os
from typing import Dict, Any, Optional, List, Tuple
import comfy.model_management as model_management

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
            # print(f"🎭 Higgs Audio: Using mode '{multi_speaker_mode}'")
            
            # Extract generation parameters from engine config to pass to all segment calls
            generation_params = {
                "temperature": self.engine_wrapper.config.get("temperature", 0.8),
                "top_p": self.engine_wrapper.config.get("top_p", 0.6),
                "top_k": self.engine_wrapper.config.get("top_k", 80),
                "max_new_tokens": self.engine_wrapper.config.get("max_new_tokens", 2048),
                "force_audio_gen": self.engine_wrapper.config.get("force_audio_gen", False),
                "ras_win_len": self.engine_wrapper.config.get("ras_win_len", 7),
                "ras_max_num_repeat": self.engine_wrapper.config.get("ras_max_num_repeat", 2),
                "system_prompt": self.engine_wrapper.config.get("system_prompt", "Generate audio following instruction.")
            }
            
            if multi_speaker_mode == "Custom Character Switching":
                # Use existing modular utilities - pause processing first, then character parsing (like ChatterBox)
                # print(f"🎭 Higgs Audio: Using character switching with pause support")

                # Import modular utilities
                from utils.text.pause_processor import PauseTagProcessor
                from utils.voice.discovery import get_character_mapping, get_available_characters
                from utils.text.character_parser import parse_character_text, character_parser

                # Set up character parser with available characters BEFORE processing text
                available_chars = get_available_characters()
                character_parser.set_available_characters(list(available_chars))

                # Discover characters and build voice mapping (include narrator for fallback)
                # Use parse_text_segments to get full segment info including parameters
                character_segment_objects = character_parser.parse_text_segments(text)
                character_segments = [(seg.character, seg.text) for seg in character_segment_objects]
                all_characters = set(char for char, _ in character_segments)
                all_characters.add("narrator")  # Always include narrator for fallback mapping
                character_mapping = get_character_mapping(list(all_characters), engine_type="f5tts")

                # Build parameter mapping for later use (map from segment text to parameters)
                segment_params_map = {seg.text: seg.parameters for seg in character_segment_objects if seg.parameters}
                
                # print(f"🎭 Higgs Audio: Processing {len(character_segments)} character segment(s) - {', '.join(sorted(all_characters))}")
                
                # Build voice references with proper fallback hierarchy for narrator:
                # 1. opt_narrator (already in audio_tensor/reference_text)
                # 2. narrator_voice dropdown (handled by unified node)  
                # 3. Mapped narrator (from character alias map - e.g., David Attenborough)
                # 4. None (if user removed narrator from mapping)
                
                narrator_voice_dict = None
                narrator_ref_text = ""
                
                if audio_tensor is not None:
                    # Priority 1 & 2: Connected narrator (from Character Voices or dropdown)
                    narrator_voice_dict = {"waveform": audio_tensor["waveform"], "sample_rate": audio_tensor["sample_rate"]}
                    narrator_ref_text = reference_text or ""
                    ref_display = f"'{narrator_ref_text[:50]}...'" if narrator_ref_text else "No reference text"
                    # print(f"📖 Using connected narrator voice (Character Voices node or dropdown) | Ref: {ref_display}")
                else:
                    # Priority 3: Check mapped narrator (e.g., David Attenborough from alias map)
                    mapped_narrator = character_mapping.get("narrator", (None, None))
                    if mapped_narrator[0] and os.path.exists(mapped_narrator[0]):
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(mapped_narrator[0])
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        narrator_voice_dict = {"waveform": waveform, "sample_rate": sample_rate}
                        narrator_ref_text = mapped_narrator[1] or ""
                        ref_display = f"'{narrator_ref_text[:50]}...'" if narrator_ref_text else "No reference text"
                        print(f"📖 Using mapped narrator voice: {mapped_narrator[0]} | Ref: {ref_display}")
                    else:
                        # Priority 4: No narrator available
                        print(f"⚠️ No narrator voice available - characters will use basic TTS when no character voice found")
                
                voice_refs = {'narrator': narrator_voice_dict}
                ref_texts = {'narrator': narrator_ref_text}

                # Check for interruption before character setup
                if model_management.interrupt_processing:
                    raise InterruptedError("Higgs Audio TTS: Character voice setup interrupted by user")

                for character in all_characters:
                    # Skip narrator - already set above with connected voice
                    if character.lower() == "narrator":
                        continue
                        
                    audio_path, char_ref_text = character_mapping.get(character, (None, None))
                    if audio_path and os.path.exists(audio_path):
                        import torchaudio
                        waveform, sample_rate = torchaudio.load(audio_path)
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        voice_refs[character] = {"waveform": waveform, "sample_rate": sample_rate}
                        
                        # Use character-specific reference text from mapping
                        if char_ref_text and char_ref_text.strip():
                            ref_texts[character] = char_ref_text.strip()
                            print(f"🎭 {character}: Loaded voice + ref text: '{char_ref_text[:30]}...'")
                        else:
                            ref_texts[character] = ""  # Empty ref text for character's own voice
                            print(f"🎭 {character}: Loaded voice, no ref text")  
                    else:
                        # Use main narrator voice and text as fallback
                        voice_refs[character] = narrator_voice_dict
                        ref_texts[character] = reference_text or ""
                        if narrator_voice_dict:
                            print(f"🎭 {character}: Using narrator voice + ref text")
                        else:
                            print(f"⚠️ {character}: No voice available, using basic TTS")
                
                def tts_generate_func(text_content: str) -> torch.Tensor:
                    """TTS generation function for pause tag processor"""
                    if '[' in text_content and ']' in text_content:
                        # Handle character switching within this segment
                        # Use parse_text_segments to get full segment info including parameters
                        char_segment_objects = character_parser.parse_text_segments(text_content)
                        segment_audio_parts = []

                        for seg in char_segment_objects:
                            character = seg.character
                            segment_text = seg.text
                            segment_params = seg.parameters if seg.parameters else {}

                            # Check for interruption during character segment processing
                            if model_management.interrupt_processing:
                                raise InterruptedError(f"Higgs Audio TTS character segment ({character}) interrupted by user")

                            char_audio_dict = voice_refs.get(character)
                            char_ref_text = ref_texts.get(character, reference_text or "")

                            # Apply per-segment parameters
                            current_params = dict(generation_params)
                            current_seed = seed
                            if segment_params:
                                from utils.text.segment_parameters import apply_segment_parameters
                                current_params = apply_segment_parameters(current_params, segment_params, "higgs_audio")
                                # If segment has a seed parameter, use that instead of the default
                                if 'seed' in current_params:
                                    current_seed = current_params.pop('seed')
                                print(f"📊 Higgs Audio segment: Character '{character}' with parameters {segment_params}")

                            segment_result = self.engine_wrapper.generate_tts_audio(
                                text=segment_text,
                                char_audio=char_audio_dict,
                                char_text=char_ref_text,
                                character=character,
                                seed=current_seed,
                                enable_audio_cache=enable_audio_cache,
                                max_chars_per_chunk=max_chars_per_chunk,
                                silence_between_chunks_ms=0,
                                **current_params  # Pass through all generation parameters including segment-specific ones
                            )
                            segment_audio_parts.append(segment_result)

                        # Combine character segments
                        if segment_audio_parts:
                            return torch.cat(segment_audio_parts, dim=-1)
                        else:
                            return torch.zeros(1, 0)
                    else:
                        # Simple text segment without character switching - use narrator voice
                        # Check if this text has parameters from the segment_params_map
                        segment_params = segment_params_map.get(text_content, {})

                        narrator_audio = voice_refs.get("narrator")
                        if narrator_audio is None and audio_tensor is not None:
                            narrator_audio = {"waveform": audio_tensor, "sample_rate": 24000}

                        # Apply per-segment parameters
                        current_params = dict(generation_params)
                        current_seed = seed
                        if segment_params:
                            from utils.text.segment_parameters import apply_segment_parameters
                            current_params = apply_segment_parameters(current_params, segment_params, "higgs_audio")
                            # If segment has a seed parameter, use that instead of the default
                            if 'seed' in current_params:
                                current_seed = current_params.pop('seed')
                            print(f"📊 Higgs Audio segment: Character 'narrator' with parameters {segment_params}")

                        return self.engine_wrapper.generate_tts_audio(
                            text=text_content,
                            char_audio=narrator_audio,
                            char_text=ref_texts.get("narrator", reference_text or ""),
                            character="narrator",
                            seed=current_seed,
                            enable_audio_cache=enable_audio_cache,
                            max_chars_per_chunk=max_chars_per_chunk,
                            silence_between_chunks_ms=0,
                            **current_params  # Pass through all generation parameters including segment-specific ones
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

                # Check for interruption before native mode processing
                if model_management.interrupt_processing:
                    raise InterruptedError("Higgs Audio TTS: Native multi-speaker mode setup interrupted by user")

                # Get second narrator audio if provided
                opt_second_narrator = self.engine_wrapper.config.get("opt_second_narrator")
                
                # Prepare reference audios for native mode
                reference_audio_dict = None
                second_audio_dict = None
                second_narrator_ref_text = ""
                
                if audio_tensor is not None:
                    reference_audio_dict = {
                        "waveform": audio_tensor,
                        "sample_rate": 24000
                    }
                
                if opt_second_narrator is not None:
                    # Extract reference text from Character Voices data structure
                    if isinstance(opt_second_narrator, dict) and "audio" in opt_second_narrator:
                        # Character Voices node output
                        second_audio_dict = opt_second_narrator.get("audio")
                        second_narrator_ref_text = opt_second_narrator.get("reference_text", "")
                        print(f"🎭 Using Character Voices reference text for second narrator: '{second_narrator_ref_text[:50]}...'")
                    elif isinstance(opt_second_narrator, dict) and "waveform" in opt_second_narrator:
                        # Direct audio input
                        second_audio_dict = opt_second_narrator
                        second_narrator_ref_text = ""
                        print(f"🎭 Using direct audio input for second narrator (no reference text)")
                    else:
                        second_audio_dict = opt_second_narrator
                
                # Process entire conversation as single unit - let Higgs Audio handle pauses and speaker transitions
                print(f"🎭 Processing full conversation: '{text[:100]}...'")
                
                result = self.engine_wrapper.generate_tts_audio(
                    text=text,  # Full conversation text
                    char_audio=reference_audio_dict,
                    char_text=reference_text or "",  # Use reference text to improve voice cloning quality
                    character="SPEAKER0",
                    seed=seed,
                    enable_audio_cache=enable_audio_cache,
                    max_chars_per_chunk=max_chars_per_chunk,  # This might trigger chunking - unknown how it will interact
                    silence_between_chunks_ms=0,
                    # Native mode specific parameters
                    multi_speaker_mode=multi_speaker_mode,
                    second_narrator_audio=second_audio_dict,
                    second_narrator_text=second_narrator_ref_text,  # Use extracted reference text from Character Voices
                    **generation_params  # Pass through all generation parameters
                )
            
            # CRITICAL FIX: Ensure tensor has correct dimensions for ComfyUI
            if isinstance(result, torch.Tensor):
                # print(f"🔧 Higgs Audio tensor before fix: {result.shape}")
                if result.dim() == 3 and result.size(0) == 1:
                    result = result.squeeze(0)  # Remove batch dimension [1,1,N] -> [1,N]
                    # print(f"🔧 Higgs Audio tensor after fix: {result.shape}")
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
                
                # print(f"🔧 Higgs Audio raw tensor: {result.shape}")
                # print(f"🔧 After format_for_comfyui: {formatted_audio['waveform'].shape}")
                
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