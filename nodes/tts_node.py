"""
ChatterBox TTS Node - Migrated to use new foundation
Enhanced Text-to-Speech node using ChatterboxTTS with improved chunking
"""

import torch
import numpy as np
import os
import gc
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load base_node module directly
base_node_path = os.path.join(current_dir, "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from core.text_chunking import ImprovedChatterBoxChunker
from core.audio_processing import AudioProcessingUtils
from core.voice_discovery import get_available_characters, get_character_mapping
from core.character_parser import parse_character_text, character_parser
import comfy.model_management as model_management


class ChatterboxTTSNode(BaseTTSNode):
    """
    Enhanced Text-to-Speech node using ChatterboxTTS - Voice Edition
    SUPPORTS BUNDLED CHATTERBOX + Enhanced Chunking + Character Switching
    Supports character switching using [Character] tags in text.
    """
    
    @classmethod
    def NAME(cls):
        return "🎤 ChatterBox Voice TTS (diogod)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": """Hello! This is enhanced ChatterboxTTS with character switching.
[Alice] Hi there! I'm Alice speaking with ChatterBox voice.
[Bob] And I'm Bob! Great to meet you both.
Back to the main narrator voice for the conclusion.""",
                    "tooltip": "Text to convert to speech. Use [Character] tags for voice switching. Characters not found in voice folders will use the main reference audio."
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "exaggeration": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.25, 
                    "max": 2.0, 
                    "step": 0.05
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.05, 
                    "max": 5.0, 
                    "step": 0.05
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "audio_prompt_path": ("STRING", {"default": ""}),
                # ENHANCED CHUNKING CONTROLS - ALL OPTIONAL FOR BACKWARD COMPATIBILITY
                "enable_chunking": ("BOOLEAN", {"default": True}),
                "max_chars_per_chunk": ("INT", {"default": 400, "min": 100, "max": 1000, "step": 50}),
                "chunk_combination_method": (["auto", "concatenate", "silence_padding", "crossfade"], {"default": "auto"}),
                "silence_between_chunks_ms": ("INT", {"default": 100, "min": 0, "max": 500, "step": 25}),
                "crash_protection_template": ("STRING", {
                    "default": "hmm ,, {seg} hmm ,,",
                    "tooltip": "Custom padding template for short text segments to prevent ChatterBox crashes. ChatterBox has a bug where text shorter than ~21 characters causes CUDA tensor errors in sequential generation. Use {seg} as placeholder for the original text. Examples: '...ummmmm {seg}' (default hesitation), '{seg}... yes... {seg}' (repetition), 'Well, {seg}' (natural prefix), or empty string to disable padding. This only affects ChatterBox nodes, not F5-TTS nodes."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        super().__init__()
        self.chunker = ImprovedChatterBoxChunker()
    
    def _pad_short_text_for_chatterbox(self, text: str, padding_template: str = "hmm ,, {seg} hmm ,,", min_length: int = 15) -> str:
        """
        Add custom padding to short text to prevent ChatterBox crashes.
        
        ChatterBox has a bug where short text segments cause CUDA tensor indexing errors
        in sequential generation scenarios. Adding meaningful tokens with custom templates
        prevents these crashes while allowing user customization.
        
        Based on testing:
        - "w" + spaces/periods crashes even with 150 char padding
        - "word is a word is a world" works for 4+ runs
        - "...ummmmm w" provides natural hesitation + preserves original text
        
        Args:
            text: Input text to check and pad if needed
            padding_template: Custom template with {seg} placeholder for original text
            min_length: Minimum text length threshold (default: 21 characters)
            
        Returns:
            Original text or text with custom padding template if too short
        """
        stripped_text = text.strip()
        if len(stripped_text) < min_length:
            # If template is empty, disable padding
            if not padding_template.strip():
                return text
            # Replace {seg} placeholder with original text
            return padding_template.replace("{seg}", stripped_text)
        return text

    def _is_problematic_text(self, text: str, is_already_padded: bool = False) -> tuple[bool, str]:
        """
        Predict if text is likely to cause ChatterBox CUDA crashes.
        Based on analysis of crash patterns.
        
        Args:
            text: The text to check (may be original or already padded)
            is_already_padded: True if text is already padded, False if it needs padding check
        
        Returns:
            tuple: (is_problematic, reason)
        """
        # Don't strip - leading/trailing spaces might help prevent the bug
        original_text = text
        
        # If text is already padded, check its length directly
        # If not padded, check what the length would be after padding
        if is_already_padded:
            final_length = len(original_text)
            display_text = repr(original_text)  # repr shows spaces clearly
        else:
            padded_text = self._pad_short_text_for_chatterbox(text)
            final_length = len(padded_text)
            display_text = f"{repr(original_text)} → padded: {repr(padded_text)}"
        
        # Text shorter than 21 characters (after padding if needed) is high risk
        if final_length < 15:
            return True, f"text too short ({final_length} chars < 21) - {display_text}"
        
        # Repetitive patterns like "Yes!Yes!Yes!" are high risk
        # if len(stripped) <= 20 and stripped.count(stripped[:4]) > 1:
        #     return True, f"repetitive pattern detected ('{stripped[:4]}' appears {stripped.count(stripped[:4])} times)"
        
        # Single words with exclamations (check the actual text, not stripped)
        text_without_spaces = original_text.replace(' ', '')
        if len(original_text.split()) == 1 and ('!' in original_text or '?' in original_text):
            return True, f"single word with punctuation ({repr(original_text)})"
        
        # Short phrases with repetitive character patterns
        if len(original_text) <= 25 and len(set(text_without_spaces)) <= 4:
            return True, f"limited character variety ({len(set(text_without_spaces))} unique chars in {len(original_text)} chars) - {repr(original_text)}"
        
        return False, ""



    def _safe_generate_tts_audio(self, text, audio_prompt, exaggeration, temperature, cfg_weight, enable_crash_protection=True):
        """
        Wrapper around generate_tts_audio with crash protection.
        If enable_crash_protection=False, behaves like original generate_tts_audio.
        """
        if not enable_crash_protection:
            # No protection - original behavior (may crash ComfyUI)
            return self.generate_tts_audio(text, audio_prompt, exaggeration, temperature, cfg_weight)
        
        # Predict and skip problematic text before it crashes
        # The text passed here is already processed/padded, so check it directly
        is_problematic, reason = self._is_problematic_text(text, is_already_padded=True)
        if is_problematic:
            print(f"🚨 SKIPPING PROBLEMATIC SEGMENT: '{text[:50]}...' - Reason: {reason}")
            print(f"🛡️ Generating silence to prevent ChatterBox CUDA crash and avoid ComfyUI reboot")
            # Return silence instead of attempting generation
            silence_duration = max(1.0, len(text) * 0.05)  # Rough estimate
            silence_samples = int(silence_duration * (self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050))
            return torch.zeros(1, silence_samples)
        
        # If prediction says it's safe, try generation with fallback
        try:
            return self.generate_tts_audio(text, audio_prompt, exaggeration, temperature, cfg_weight)
        except Exception as e:
            error_msg = str(e)
            is_cuda_crash = ("srcIndex < srcSelectDimSize" in error_msg or 
                           "CUDA" in error_msg or 
                           "device-side assert" in error_msg or
                           "an illegal memory access" in error_msg)
            if is_cuda_crash:
                print(f"🚨 UNEXPECTED CUDA CRASH occurred during generation: '{text[:50]}...'")
                print(f"🛡️ Crash detection missed this pattern - returning silence to prevent ComfyUI reboot")
                # Return silence instead of crashing
                silence_duration = max(1.0, len(text) * 0.05)  # Rough estimate
                silence_samples = int(silence_duration * (self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050))
                return torch.zeros(1, silence_samples)
            else:
                raise


    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate and normalize inputs."""
        validated = super().validate_inputs(**inputs)
        
        # Handle None/empty values for backward compatibility
        if validated.get("enable_chunking") is None:
            validated["enable_chunking"] = True
        if validated.get("max_chars_per_chunk") is None or validated.get("max_chars_per_chunk", 0) < 100:
            validated["max_chars_per_chunk"] = 400
        if not validated.get("chunk_combination_method"):
            validated["chunk_combination_method"] = "auto"
        if validated.get("silence_between_chunks_ms") is None:
            validated["silence_between_chunks_ms"] = 100
        if validated.get("crash_protection_template") is None:
            validated["crash_protection_template"] = "hmm ,, {seg} hmm ,,"
        
        return validated

    def combine_audio_chunks(self, audio_segments: List[torch.Tensor], method: str, 
                           silence_ms: int, text_length: int) -> torch.Tensor:
        """Combine audio segments using specified method."""
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Auto-select best method based on text length
        if method == "auto":
            if text_length > 1000:  # Very long text
                method = "silence_padding"
            elif text_length > 500:  # Medium text
                method = "crossfade"
            else:  # Short text
                method = "concatenate"
        
        if method == "concatenate":
            return AudioProcessingUtils.concatenate_audio_segments(audio_segments, "simple")
        
        elif method == "silence_padding":
            silence_duration = silence_ms / 1000.0  # Convert to seconds
            return AudioProcessingUtils.concatenate_audio_segments(
                audio_segments, "silence", silence_duration=silence_duration, 
                sample_rate=self.tts_model.sr
            )
        
        elif method == "crossfade":
            return AudioProcessingUtils.concatenate_audio_segments(
                audio_segments, "crossfade", crossfade_duration=0.1, 
                sample_rate=self.tts_model.sr
            )
        
        else:
            # Fallback to concatenation
            return AudioProcessingUtils.concatenate_audio_segments(audio_segments, "simple")

    def generate_speech(self, text, device, exaggeration, temperature, cfg_weight, seed, 
                       reference_audio=None, audio_prompt_path="", 
                       enable_chunking=True, max_chars_per_chunk=400, 
                       chunk_combination_method="auto", silence_between_chunks_ms=100,
                       crash_protection_template="hmm ,, {seg} hmm ,,", 
                       enable_crash_protection=True):
        
        def _process():
            # Validate inputs
            inputs = self.validate_inputs(
                text=text, device=device, exaggeration=exaggeration,
                temperature=temperature, cfg_weight=cfg_weight, seed=seed,
                reference_audio=reference_audio, audio_prompt_path=audio_prompt_path,
                enable_chunking=enable_chunking, max_chars_per_chunk=max_chars_per_chunk,
                chunk_combination_method=chunk_combination_method,
                silence_between_chunks_ms=silence_between_chunks_ms,
                crash_protection_template=crash_protection_template,
                enable_crash_protection=enable_crash_protection
            )
            
            # Load model
            self.load_tts_model(inputs["device"])
            
            # Set seed for reproducibility
            self.set_seed(inputs["seed"])
            
            # Handle main reference audio
            main_audio_prompt = self.handle_reference_audio(
                inputs.get("reference_audio"), inputs.get("audio_prompt_path", "")
            )
            
            # Parse character segments from text
            character_segments = parse_character_text(inputs["text"])
            
            # Check if we have character switching
            characters = list(set(char for char, _ in character_segments))
            has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
            
            if has_multiple_characters:
                # CHARACTER SWITCHING MODE
                print(f"🎭 ChatterBox: Character switching mode - found characters: {', '.join(characters)}")
                
                # Set up character parser with available characters
                available_chars = get_available_characters()
                character_parser.set_available_characters(list(available_chars))
                
                # Get character voice mapping (ChatterBox doesn't need reference text)
                character_mapping = get_character_mapping(characters, engine_type="chatterbox")
                
                # Build voice references with fallback to main voice
                voice_refs = {}
                for character in characters:
                    audio_path, _ = character_mapping.get(character, (None, None))
                    if audio_path:
                        voice_refs[character] = audio_path
                        print(f"🎭 Using character voice for '{character}'")
                    else:
                        voice_refs[character] = main_audio_prompt
                        print(f"🔄 Using main voice for character '{character}' (not found in voice folders)")
                
                # Process each character segment
                audio_segments = []
                for i, (character, segment_text) in enumerate(character_segments):
                    # Check for interruption
                    self.check_interruption(f"ChatterBox character generation segment {i+1}/{len(character_segments)}")
                    
                    # Apply chunking to long segments if enabled (PRESERVE EXISTING CHUNKING)
                    if inputs["enable_chunking"] and len(segment_text) > inputs["max_chars_per_chunk"]:
                        segment_chunks = self.chunker.split_into_chunks(segment_text, inputs["max_chars_per_chunk"])
                    else:
                        segment_chunks = [segment_text]
                    
                    # Get voice reference for this character
                    char_audio_prompt = voice_refs[character]
                    
                    # Generate audio for each chunk of this character segment (PRESERVE EXISTING TTS GENERATION)
                    for chunk_i, chunk_text in enumerate(segment_chunks):
                        print(f"🎤 Generating ChatterBox segment {i+1}/{len(character_segments)} chunk {chunk_i+1}/{len(segment_chunks)} for '{character}'...")
                        
                        # BUGFIX: Pad short text with custom template to prevent ChatterBox sequential generation crashes
                        # Only for ChatterBox (not F5TTS) and only when text is very short
                        processed_chunk_text = self._pad_short_text_for_chatterbox(chunk_text, inputs["crash_protection_template"])
                        
                        # DEBUG: Show actual text being sent to ChatterBox when padding might occur
                        if len(chunk_text.strip()) < 21:  # Show for all segments at or below padding threshold (matches min_length in _pad_short_text_for_chatterbox)
                            print(f"🔍 DEBUG: Original text: '{chunk_text}' → Processed: '{processed_chunk_text}' (len: {len(processed_chunk_text)})")
                        
                        try:
                            # Determine crash protection based on padding template
                            enable_protection = bool(inputs["crash_protection_template"].strip())
                            chunk_audio = self._safe_generate_tts_audio(
                                processed_chunk_text, char_audio_prompt, inputs["exaggeration"], 
                                inputs["temperature"], inputs["cfg_weight"], enable_protection
                            )
                            print(f"✅ ChatterBox segment {i+1} chunk {chunk_i+1} completed successfully")
                            print(f"🔍 DEBUG: Audio shape: {chunk_audio.shape}, dtype: {chunk_audio.dtype}")
                            audio_segments.append(chunk_audio)
                            print(f"🔍 DEBUG: Total segments so far: {len(audio_segments)}")
                        except Exception as e:
                            print(f"❌ ChatterBox segment {i+1} chunk {chunk_i+1} failed: {e}")
                            raise
                
                # Combine all character segments (PRESERVE EXISTING COMBINE LOGIC)
                wav = self.combine_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], len(inputs["text"])
                )
                
                # Generate info
                total_duration = wav.size(-1) / self.tts_model.sr
                model_source = self.model_manager.get_model_source("tts")
                info = f"Generated {total_duration:.1f}s character-switched audio from {len(character_segments)} segments using {len(characters)} characters ({model_source} models)"
                
            else:
                # SINGLE CHARACTER MODE (PRESERVE ORIGINAL BEHAVIOR)
                text_length = len(inputs["text"])
                
                if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                    # Process single chunk with crash protection
                    # Determine crash protection based on padding template
                    enable_protection = bool(inputs["crash_protection_template"].strip())
                    wav = self._safe_generate_tts_audio(
                        inputs["text"], main_audio_prompt, inputs["exaggeration"], 
                        inputs["temperature"], inputs["cfg_weight"], enable_protection
                    )
                    model_source = self.model_manager.get_model_source("tts")
                    info = f"Generated {wav.size(-1) / self.tts_model.sr:.1f}s audio from {text_length} characters (single chunk, {model_source} models)"
                else:
                    # Split into chunks using improved chunker (UNCHANGED)
                    chunks = self.chunker.split_into_chunks(inputs["text"], inputs["max_chars_per_chunk"])
                    
                    # Process each chunk (UNCHANGED)
                    audio_segments = []
                    for i, chunk in enumerate(chunks):
                        # Check for interruption
                        self.check_interruption(f"TTS generation chunk {i+1}/{len(chunks)}")
                        
                        # Show progress for multi-chunk generation
                        print(f"🎤 Generating TTS chunk {i+1}/{len(chunks)}...")
                        
                        # Determine crash protection based on padding template
                        enable_protection = bool(inputs["crash_protection_template"].strip())
                        chunk_audio = self._safe_generate_tts_audio(
                            chunk, main_audio_prompt, inputs["exaggeration"], 
                            inputs["temperature"], inputs["cfg_weight"], enable_protection
                        )
                        audio_segments.append(chunk_audio)
                    
                    # Combine audio segments (UNCHANGED)
                    wav = self.combine_audio_chunks(
                        audio_segments, inputs["chunk_combination_method"], 
                        inputs["silence_between_chunks_ms"], text_length
                    )
                    
                    # Generate info (UNCHANGED)
                    total_duration = wav.size(-1) / self.tts_model.sr
                    avg_chunk_size = text_length // len(chunks)
                    model_source = self.model_manager.get_model_source("tts")
                    info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, {model_source} models)"
            
            # Return audio in ComfyUI format
            return (
                self.format_audio_output(wav, self.tts_model.sr),
                info
            )
        
        return self.process_with_error_handling(_process)