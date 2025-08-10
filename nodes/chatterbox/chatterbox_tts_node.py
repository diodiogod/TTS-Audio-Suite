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
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_characters, get_character_mapping
from utils.text.pause_processor import PauseTagProcessor
from utils.text.character_parser import parse_character_text, character_parser
import comfy.model_management as model_management

# Global audio cache for ChatterBox TTS segments
GLOBAL_AUDIO_CACHE = {}


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
        # Import language models for dropdown
        try:
            from engines.chatterbox.language_models import get_available_languages
            available_languages = get_available_languages()
        except ImportError:
            available_languages = ["English"]
        
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
                "language": (available_languages, {
                    "default": "English",
                    "tooltip": "Language model to use for text-to-speech generation. Local models are preferred over remote downloads."
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
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, generated audio segments will be cached in memory to speed up subsequent runs with identical parameters."
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Number of workers for parallel processing. 0-1 = Sequential processing, 2+ = Continuous parallel streaming. Higher values = faster generation but more memory usage."
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
        
        # BUGFIX: Don't pad text that contains only pause tags - they should be processed by PauseTagProcessor
        import re
        pause_pattern = r'\[(pause|wait|stop):(\d+(?:\.\d+)?)(s|ms)?\]'
        if re.search(pause_pattern, stripped_text):
            # Check if text contains ONLY pause tags and whitespace
            text_without_pauses = re.sub(pause_pattern, '', stripped_text).strip()
            if not text_without_pauses:
                # print(f"🚫 Skipping crash protection padding for pause-only content: '{stripped_text}'")
                return text
        
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

    def _generate_with_pause_tags(self, pause_segments: List, inputs: Dict, main_audio_prompt) -> torch.Tensor:
        """
        Generate audio with pause tag support, handling character switching within segments.
        
        Args:
            pause_segments: List of ('text', content) or ('pause', duration) segments
            inputs: Input parameters dictionary
            main_audio_prompt: Default audio prompt
            
        Returns:
            Combined audio tensor with pauses
        """
        def generate_segment_audio(segment_text: str, audio_prompt) -> torch.Tensor:
            """Generate audio for a text segment with crash protection"""
            # Apply padding for crash protection
            processed_text = self._pad_short_text_for_chatterbox(segment_text, inputs["crash_protection_template"])
            
            # Determine crash protection based on template
            enable_protection = bool(inputs["crash_protection_template"].strip())
            
            return self._safe_generate_tts_audio(
                processed_text, audio_prompt, inputs["exaggeration"], 
                inputs["temperature"], inputs["cfg_weight"], enable_protection
            )
        
        # Check if we need character switching within pause segments
        has_character_switching = any(
            segment_type == 'text' and '[' in content and ']' in content 
            for segment_type, content in pause_segments
        )
        
        if has_character_switching:
            # Set up character voice mapping
            from utils.voice.discovery import get_character_mapping
            
            # Process each segment and extract characters
            all_characters = set()
            for segment_type, content in pause_segments:
                if segment_type == 'text':
                    char_segments = parse_character_text(content)
                    chars = set(char for char, _ in char_segments)
                    all_characters.update(chars)
            
            character_mapping = get_character_mapping(list(all_characters), engine_type="chatterbox")
            
            # Build voice references
            voice_refs = {}
            for character in all_characters:
                audio_path, _ = character_mapping.get(character, (None, None))
                voice_refs[character] = audio_path if audio_path else main_audio_prompt
        
        # Generate audio using pause tag processor
        def tts_generate_func(text_content: str) -> torch.Tensor:
            """TTS generation function for pause tag processor"""
            if has_character_switching and ('[' in text_content and ']' in text_content):
                # Handle character switching within this segment
                char_segments = parse_character_text(text_content)
                segment_audio_parts = []
                
                for character, segment_text in char_segments:
                    audio_prompt = voice_refs.get(character, main_audio_prompt)
                    audio_part = generate_segment_audio(segment_text, audio_prompt)
                    segment_audio_parts.append(audio_part)
                
                # Combine character segments
                if segment_audio_parts:
                    return torch.cat(segment_audio_parts, dim=-1)
                else:
                    return torch.zeros(1, 0)
            else:
                # Simple text segment without character switching
                return generate_segment_audio(text_content, main_audio_prompt)
        
        return PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, tts_generate_func, self.tts_model.sr
        )

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
            sr = self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050
            return AudioProcessingUtils.concatenate_audio_segments(
                audio_segments, "silence", silence_duration=silence_duration, 
                sample_rate=sr
            )
        
        elif method == "crossfade":
            sr = self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050
            return AudioProcessingUtils.concatenate_audio_segments(
                audio_segments, "crossfade", crossfade_duration=0.1, 
                sample_rate=sr
            )
        
        else:
            # Fallback to concatenation
            return AudioProcessingUtils.concatenate_audio_segments(audio_segments, "simple")
    
    def _generate_stable_audio_component(self, reference_audio, audio_prompt_path: str) -> str:
        """Generate stable identifier for audio prompt to prevent cache invalidation from temp file paths."""
        if reference_audio is not None:
            waveform_hash = hashlib.md5(reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
            return f"ref_audio_{waveform_hash}_{reference_audio['sample_rate']}"
        elif audio_prompt_path:
            return audio_prompt_path
        else:
            return ""

    def _generate_segment_cache_key(self, text: str, exaggeration: float, temperature: float, 
                                   cfg_weight: float, seed: int, audio_component: str, 
                                   model_source: str, device: str, language: str = "English", 
                                   character: str = "narrator") -> str:
        """Generate cache key for a single audio segment based on generation parameters."""
        cache_data = {
            'text': text,
            'exaggeration': exaggeration,
            'temperature': temperature,
            'cfg_weight': cfg_weight,
            'seed': seed,
            'audio_component': audio_component,
            'model_source': model_source,
            'device': device,
            'language': language,
            'character': character,
            'engine': 'chatterbox'
        }
        cache_string = str(sorted(cache_data.items()))
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_key

    def _get_cached_segment_audio(self, segment_cache_key: str) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve cached audio for a single segment if available from global cache."""
        return GLOBAL_AUDIO_CACHE.get(segment_cache_key)

    def _cache_segment_audio(self, segment_cache_key: str, audio_tensor: torch.Tensor, natural_duration: float):
        """Cache generated audio for a single segment in global cache."""
        GLOBAL_AUDIO_CACHE[segment_cache_key] = (audio_tensor.clone(), natural_duration)

    def _generate_tts_with_pause_tags(self, text: str, audio_prompt, exaggeration: float, 
                                    temperature: float, cfg_weight: float, language: str = "English",
                                    enable_pause_tags: bool = True, character: str = "narrator", 
                                    seed: int = 0, enable_cache: bool = True,
                                    crash_protection_template: str = "hmm ,, {seg} hmm ,,", 
                                    stable_audio_component: str = None) -> torch.Tensor:
        """
        Generate ChatterBox TTS audio with pause tag support.
        
        Args:
            text: Input text potentially with pause tags
            audio_prompt: Audio prompt for TTS generation
            exaggeration: ChatterBox exaggeration parameter
            temperature: ChatterBox temperature parameter
            cfg_weight: ChatterBox CFG weight parameter
            enable_pause_tags: Whether to process pause tags
            character: Character name for cache key
            seed: Seed for reproducibility and cache key
            enable_cache: Whether to use caching
            crash_protection_template: Template for crash protection
            stable_audio_component: Stable audio identifier for cache
            
        Returns:
            Generated audio tensor with pauses
        """
        # Preprocess text for pause tags
        processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
            text, enable_pause_tags
        )
        
        if pause_segments is None:
            # No pause tags, use regular generation with caching
            if enable_cache:
                # Use stable audio component for cache key
                audio_component = stable_audio_component if stable_audio_component else ""
                
                # Apply crash protection first for consistency
                protected_text = self._pad_short_text_for_chatterbox(processed_text, crash_protection_template)
                
                # Use language-based model source for consistent cache keys across streaming/traditional
                model_source = f"chatterbox_{language.lower()}"
                cache_key = self._generate_segment_cache_key(
                    protected_text, exaggeration, temperature, cfg_weight, seed,
                    audio_component, model_source, self.device, language, character
                )
                
                # Try cache first
                # print(f"🔍 STREAMING cache: text='{protected_text}', char='{character}', lang='{language}', model='{model_source}', seed={seed}, exag={exaggeration}, temp={temperature}, cfg={cfg_weight}, device='{self.device}', audio='{audio_component}'")
                cached_data = self._get_cached_segment_audio(cache_key)
                if cached_data:
                    print(f"💾 CACHE HIT for {character}: '{processed_text[:30]}...'")
                    return cached_data[0]
                # else:
                #     print(f"❌ CACHE MISS for {character}: '{processed_text[:30]}...'")
                
                # Generate and cache
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
                # Clone tensor to avoid autograd issues in streaming mode
                audio_clone = audio.detach().clone() if audio.requires_grad else audio
                self._cache_segment_audio(cache_key, audio_clone, 0.0)  # Duration not needed for basic caching
                return audio_clone
            else:
                protected_text = self._pad_short_text_for_chatterbox(processed_text, crash_protection_template)
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
                # Clone tensor to avoid autograd issues in streaming mode
                return audio.detach().clone() if audio.requires_grad else audio
        
        # Generate audio with pause tags, caching individual text segments
        def tts_generate_func(text_content: str) -> torch.Tensor:
            """TTS generation function for pause tag processor with caching"""
            if enable_cache:
                # Use stable audio component for cache key
                audio_component = stable_audio_component if stable_audio_component else ""
                
                # Apply crash protection first for consistency
                protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                # Use language-based model source for consistent cache keys across streaming/traditional
                model_source = f"chatterbox_{language.lower()}"
                cache_key = self._generate_segment_cache_key(
                    protected_text, exaggeration, temperature, cfg_weight, seed,
                    audio_component, model_source, self.device, language, character
                )
                
                # Try cache first  
                # print(f"🔍 TRADITIONAL cache: text='{protected_text}', char='{character}', lang='{language}', model='{model_source}', seed={seed}, exag={exaggeration}, temp={temperature}, cfg={cfg_weight}, device='{self.device}', audio='{audio_component}'")
                cached_data = self._get_cached_segment_audio(cache_key)
                if cached_data:
                    print(f"💾 CACHE HIT for {character}: '{text_content[:30]}...'")
                    return cached_data[0]
                # else:
                #     print(f"❌ CACHE MISS for {character}: '{text_content[:30]}...'")
                
                # Generate and cache
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
                # Clone tensor to avoid autograd issues in streaming mode
                audio_clone = audio.detach().clone() if audio.requires_grad else audio
                self._cache_segment_audio(cache_key, audio_clone, 0.0)  # Duration not needed for basic caching
                return audio_clone
            else:
                # Apply crash protection
                protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight)
                # Clone tensor to avoid autograd issues in streaming mode
                return audio.detach().clone() if audio.requires_grad else audio
        
        return PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, tts_generate_func, self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050
        )

    def _generate_with_pause_tags(self, pause_segments, inputs, main_audio_prompt):
        """Generate audio using the pause tag processor with character switching support."""
        if inputs.get("enable_audio_cache"):
            stable_audio_component = self._generate_stable_audio_component(
                inputs.get("reference_audio"), inputs.get("audio_prompt_path", "")
            )
        else:
            stable_audio_component = ""
        
        # Use the pause tag processor with caching
        return self._generate_tts_with_pause_tags(
            inputs["text"], main_audio_prompt, inputs["exaggeration"], 
            inputs["temperature"], inputs["cfg_weight"], inputs["language"],
            True, character="narrator", seed=inputs["seed"], 
            enable_cache=inputs.get("enable_audio_cache", True),
            crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
            stable_audio_component=stable_audio_component
        )

    def generate_speech(self, text, language, device, exaggeration, temperature, cfg_weight, seed,
                       reference_audio=None, audio_prompt_path="",
                       enable_chunking=True, max_chars_per_chunk=400,
                       chunk_combination_method="auto", silence_between_chunks_ms=100,
                       crash_protection_template="hmm ,, {seg} hmm ,,", enable_audio_cache=True,
                       batch_size=4):
        
        def _process():
            # Import PauseTagProcessor at the top to avoid scoping issues
            from utils.text.pause_processor import PauseTagProcessor
            
            # Capture batch_size from outer scope to avoid UnboundLocalError
            current_batch_size = batch_size
            
            # Validate inputs
            inputs = self.validate_inputs(
                text=text, language=language, device=device, exaggeration=exaggeration,
                temperature=temperature, cfg_weight=cfg_weight, seed=seed,
                reference_audio=reference_audio, audio_prompt_path=audio_prompt_path,
                enable_chunking=enable_chunking, max_chars_per_chunk=max_chars_per_chunk,
                chunk_combination_method=chunk_combination_method,
                silence_between_chunks_ms=silence_between_chunks_ms,
                crash_protection_template=crash_protection_template,
                enable_audio_cache=enable_audio_cache,
                batch_size=current_batch_size
            )
            
            # Set seed for reproducibility (can be done without loading model)
            self.set_seed(inputs["seed"])
            
            # Handle main reference audio
            main_audio_prompt = self.handle_reference_audio(
                inputs.get("reference_audio"), inputs.get("audio_prompt_path", "")
            )
            
            # Generate stable audio component for cache consistency
            stable_audio_component = self._generate_stable_audio_component(
                inputs.get("reference_audio"), inputs.get("audio_prompt_path", "")
            )
            
            # Preprocess text for pause tags
            processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
                inputs["text"], True
            )
            
            # Set up character parser with available characters BEFORE parsing
            from utils.voice.discovery import get_available_characters
            available_chars = get_available_characters()
            character_parser.set_available_characters(list(available_chars))
            
            # Reset session cache to allow fresh logging for new generation
            character_parser.reset_session_cache()
            
            # Set engine-aware default language to prevent unnecessary model switching
            character_parser.set_engine_aware_default_language(inputs["language"], "chatterbox")
            
            # Parse character segments from text with language awareness (use ORIGINAL text to preserve line structure)
            # NOTE: We parse characters and languages from original text, then handle pause tags within each segment
            character_segments_with_lang = character_parser.split_by_character_with_language(inputs["text"])
            
            # Check if we have pause tags, character switching, or language switching
            has_pause_tags = pause_segments is not None
            characters = list(set(char for char, _, _ in character_segments_with_lang))
            languages = list(set(lang for _, _, lang in character_segments_with_lang))
            has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
            has_multiple_languages = len(languages) > 1
            
            # Create backward-compatible character segments for existing logic
            character_segments = [(char, segment_text) for char, segment_text, _ in character_segments_with_lang]
            
            if has_multiple_characters or has_multiple_languages:
                # CHARACTER AND/OR LANGUAGE SWITCHING MODE
                if has_multiple_languages:
                    print(f"🌍 ChatterBox: Language switching mode - found languages: {', '.join(languages)}")
                if has_multiple_characters:
                    print(f"🎭 ChatterBox: Character switching mode - found characters: {', '.join(characters)}")
                
                
                # Get character voice mapping (ChatterBox doesn't need reference text)
                character_mapping = get_character_mapping(characters, engine_type="chatterbox")
                
                # Build voice references with fallback to main voice
                voice_refs = {}
                character_voices = []
                main_voices = []
                
                for character in characters:
                    audio_path, _ = character_mapping.get(character, (None, None))
                    if audio_path:
                        voice_refs[character] = audio_path
                        character_voices.append(character)
                        
                        # CRITICAL FIX: Also map resolved character name to same audio path
                        # This ensures streaming workers can find voices using resolved names
                        from utils.voice.discovery import voice_discovery
                        resolved_name = voice_discovery.resolve_character_alias(character)
                        if resolved_name != character:
                            voice_refs[resolved_name] = audio_path
                            
                    else:
                        voice_refs[character] = main_audio_prompt
                        main_voices.append(character)
                
                # Consolidated voice summary logging
                voice_summary = []
                if character_voices:
                    voice_summary.append(f"character voices: {', '.join(character_voices)}")
                if main_voices:
                    voice_summary.append(f"main voice: {', '.join(main_voices)}")
                
                if voice_summary:
                    print(f"🎭 Voice mapping - {' | '.join(voice_summary)}")
                
                # Map language codes to ChatterBox model names
                def get_chatterbox_model_for_language(lang_code: str) -> str:
                    """Map language codes to ChatterBox model names"""
                    lang_model_map = {
                        'en': 'English',          # English (always use English model)
                        'de': 'German',           # German
                        'es': 'Spanish',          # Spanish
                        'fr': 'French',           # French
                        'it': 'Italian',          # Italian
                        'pt': 'Portuguese',       # Portuguese
                        'pt-br': 'Portuguese',    # Brazilian Portuguese (use Portuguese model)
                        'pt-pt': 'Portuguese',    # European Portuguese (use Portuguese model)
                        'no': 'Norwegian',        # Norwegian
                        'nb': 'Norwegian',        # Norwegian Bokmål
                        'nn': 'Norwegian',        # Norwegian Nynorsk
                    }
                    # For the main model language, use the selected model; for others, use language-specific models
                    selected_lang = inputs["language"].lower()
                    if lang_code.lower() == selected_lang:
                        return inputs["language"]  # Use selected model for main language
                    else:
                        return lang_model_map.get(lang_code.lower(), inputs["language"])
                
                # Group segments by language with original order tracking
                language_groups = {}
                for idx, (char, segment_text, lang) in enumerate(character_segments_with_lang):
                    if lang not in language_groups:
                        language_groups[lang] = []
                    language_groups[lang].append((idx, char, segment_text, lang))  # Include original index
                
                # Generate audio for each language group, tracking original positions
                audio_segments_with_order = []  # Will store (original_index, audio_tensor)
                total_segments = len(character_segments_with_lang)
                
                # Choose processing method based on batch_size: 0-1 = sequential, 2+ = streaming
                current_batch_size = inputs.get("batch_size", 1)
                use_streaming = (current_batch_size > 1)  # Let user decide: batch_size > 1 = streaming, regardless of segment count
                
                if use_streaming:
                    # Pre-load ALL language models for streaming efficiency (prevents worker conflicts)
                    print(f"🚀 STREAMING: Pre-loading models for {len(language_groups)} languages")
                    self._preload_language_models(language_groups.keys(), inputs["device"])
                    
                    audio_segments_with_order = self._process_languages_streaming(
                        language_groups, voice_refs, inputs, character_segments_with_lang
                    )
                else:
                    audio_segments_with_order = self._process_languages_traditional(
                        language_groups, voice_refs, inputs, character_segments_with_lang
                    )
                
                # Continue to sorting and combining...
                audio_segments_with_order.sort(key=lambda x: x[0])  # Sort by original index
                audio_segments = [audio for _, audio in audio_segments_with_order]  # Extract audio tensors
                
                # Combine all character segments (PRESERVE EXISTING COMBINE LOGIC)
                wav = self.combine_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], len(inputs["text"])
                )
                
                # Generate info
                total_duration = wav.size(-1) / self.tts_model.sr
                model_source = f"chatterbox_{language.lower()}"
                
                language_info = ""
                if has_multiple_languages:
                    language_info = f" across {len(languages)} languages ({', '.join(languages)})"
                
                info = f"Generated {total_duration:.1f}s audio from {len(character_segments)} segments using {len(characters)} characters{language_info} ({model_source} models)"
                
            elif has_pause_tags:
                # PAUSE TAG MODE - handle pause tags without character/language switching
                print(f"⏸️ ChatterBox: Pause tag mode detected")
                
                # Check if pause tag segments are cached before loading model
                pause_content_cached = False
                if inputs.get("enable_audio_cache", True):
                    pause_content_cached = True
                    for segment_type, content in pause_segments:
                        if segment_type == 'text':  # Only check text segments, not pause segments
                            # Use language-based model source for consistent cache keys across streaming/traditional
                            model_source = f"chatterbox_{language.lower()}"
                            cache_key = self._generate_segment_cache_key(
                                f"narrator:{content}", inputs["exaggeration"], inputs["temperature"],
                                inputs["cfg_weight"], inputs["seed"], stable_audio_component,
                                model_source, inputs["device"], inputs["language"], "narrator"
                            )
                            cached_data = self._get_cached_segment_audio(cache_key)
                            if not cached_data:
                                pause_content_cached = False
                                break
                
                # Only load model if we need to generate something
                if not pause_content_cached:
                    print(f"🔄 Loading ChatterBox model '{inputs['language']}' for pause tag generation")
                    self.load_tts_model(inputs["device"], inputs["language"])
                else:
                    print(f"💾 All pause tag content cached - skipping model loading")
                
                # Generate audio with pause tags using special processor
                wav = self._generate_with_pause_tags(pause_segments, inputs, main_audio_prompt)
                model_source = f"chatterbox_{language.lower()}"
                
                info = f"Generated audio with pause tags (narrator voice, {model_source} models)"
                
            else:
                # SINGLE CHARACTER MODE (PRESERVE ORIGINAL BEHAVIOR)
                text_length = len(inputs["text"])
                
                # Check if single character content is cached before loading model
                single_content_cached = False
                if inputs.get("enable_audio_cache", True):
                    if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                        # Check cache for single chunk
                        processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(inputs["text"], True)
                        
                        if pause_segments is None:
                            cache_texts = [processed_text]
                        else:
                            cache_texts = [content for segment_type, content in pause_segments if segment_type == 'text']
                        
                        single_content_cached = True
                        for cache_text in cache_texts:
                            # Use language-based model source for consistent cache keys across streaming/traditional
                            model_source = f"chatterbox_{language.lower()}"
                            cache_key = self._generate_segment_cache_key(
                                f"narrator:{cache_text}", inputs["exaggeration"], inputs["temperature"],
                                inputs["cfg_weight"], inputs["seed"], stable_audio_component,
                                model_source, inputs["device"], inputs["language"], "narrator"
                            )
                            cached_data = self._get_cached_segment_audio(cache_key)
                            if not cached_data:
                                single_content_cached = False
                                break
                    else:
                        # Check cache for multiple chunks
                        chunks = self.chunker.split_into_chunks(inputs["text"], inputs["max_chars_per_chunk"])
                        single_content_cached = True
                        for chunk in chunks:
                            processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(chunk, True)
                            
                            if pause_segments is None:
                                cache_texts = [processed_text]
                            else:
                                cache_texts = [content for segment_type, content in pause_segments if segment_type == 'text']
                            
                            for cache_text in cache_texts:
                                # Get model source safely
                                model_source = f"chatterbox_{language.lower()}"
                                cache_key = self._generate_segment_cache_key(
                                    f"narrator:{cache_text}", inputs["exaggeration"], inputs["temperature"],
                                    inputs["cfg_weight"], inputs["seed"], stable_audio_component,
                                    model_source, inputs["device"], inputs["language"], "narrator"
                                )
                                cached_data = self._get_cached_segment_audio(cache_key)
                                if not cached_data:
                                    single_content_cached = False
                                    break
                            if not single_content_cached:
                                break
                
                # Only load model if we need to generate something
                if not single_content_cached:
                    print(f"🔄 Loading ChatterBox model '{inputs['language']}' for single character generation")
                    self.load_tts_model(inputs["device"], inputs["language"])
                else:
                    print(f"💾 All single character content cached - skipping model loading")
                
                if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                    # Process single chunk with caching support
                    # BUGFIX: Clean character tags from text even in single character mode
                    clean_text = character_parser.remove_character_tags(inputs["text"])
                    wav = self._generate_tts_with_pause_tags(
                        clean_text, main_audio_prompt, inputs["exaggeration"], 
                        inputs["temperature"], inputs["cfg_weight"], inputs["language"],
                        True, character="narrator", seed=inputs["seed"], 
                        enable_cache=inputs.get("enable_audio_cache", True),
                        crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                        stable_audio_component=stable_audio_component
                    )
                    model_source = f"chatterbox_{language.lower()}"
                    info = f"Generated {wav.size(-1) / self.tts_model.sr:.1f}s audio from {text_length} characters (single chunk, {model_source} models)"
                else:
                    # Split into chunks using improved chunker (UNCHANGED)
                    # BUGFIX: Clean character tags from text before chunking in single character mode
                    clean_text = character_parser.remove_character_tags(inputs["text"])
                    chunks = self.chunker.split_into_chunks(clean_text, inputs["max_chars_per_chunk"])
                    
                    # Process each chunk (UNCHANGED)
                    audio_segments = []
                    for i, chunk in enumerate(chunks):
                        # Check for interruption
                        self.check_interruption(f"TTS generation chunk {i+1}/{len(chunks)}")
                        
                        # Show progress for multi-chunk generation
                        print(f"🎤 Generating ChatterBox chunk {i+1}/{len(chunks)}...")
                        
                        # Generate chunk with caching support
                        chunk_audio = self._generate_tts_with_pause_tags(
                            chunk, main_audio_prompt, inputs["exaggeration"], 
                            inputs["temperature"], inputs["cfg_weight"], inputs["language"],
                            True, character="narrator", seed=inputs["seed"], 
                            enable_cache=inputs.get("enable_audio_cache", True),
                            crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                            stable_audio_component=stable_audio_component
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
                    model_source = f"chatterbox_{language.lower()}"
                    info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, {model_source} models)"
            
            # Return audio in ComfyUI format
            sr = self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 22050
            return (
                self.format_audio_output(wav, sr),
                info
            )
        
        return self.process_with_error_handling(_process)
    
    def _process_segment_sequential(self, original_idx, character, segment_text, segment_lang, 
                                   inputs, voice_refs, required_language, total_segments, 
                                   lang_code, stable_audio_component, audio_segments_with_order):
        """
        Helper method: Process a single segment sequentially (fallback from batch processing)
        """
        segment_display_idx = original_idx + 1  # For display (1-based)
        
        # Check for interruption
        self.check_interruption(f"ChatterBox generation segment {segment_display_idx}/{total_segments} (lang: {lang_code})")
        
        # Apply chunking to long segments if enabled
        if inputs["enable_chunking"] and len(segment_text) > inputs["max_chars_per_chunk"]:
            segment_chunks = self.chunker.split_into_chunks(segment_text, inputs["max_chars_per_chunk"])
        else:
            segment_chunks = [segment_text]
        
        # Get voice reference for this character
        char_audio_prompt = voice_refs[character]
        
        # Sequential processing for all chunks
        segment_audio_chunks = []
        for chunk_i, chunk_text in enumerate(segment_chunks):
            print(f"🎤 Generating ChatterBox segment {segment_display_idx}/{total_segments} chunk {chunk_i+1}/{len(segment_chunks)} for '{character}' (lang: {lang_code})...")
            
            # Generate audio with caching support for character segments
            chunk_audio = self._generate_tts_with_pause_tags(
                chunk_text, char_audio_prompt, inputs["exaggeration"],
                inputs["temperature"], inputs["cfg_weight"], required_language,
                True, character=character, seed=inputs.get("seed", 42),
                enable_cache=inputs.get("enable_audio_cache", True),
                crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=stable_audio_component
            )
            segment_audio_chunks.append(chunk_audio)
        
        # Combine chunks for this segment and store with original order
        if segment_audio_chunks:
            if len(segment_audio_chunks) == 1:
                segment_audio = segment_audio_chunks[0]
            else:
                segment_audio = torch.cat(segment_audio_chunks, dim=-1)
            audio_segments_with_order.append((original_idx, segment_audio))

    def _process_languages_streaming(self, language_groups, voice_refs, inputs, character_segments_with_lang):
        """Process all languages using streaming continuous workers (truly cross-character)."""
        print(f"🌊 STREAMING MODE: Processing all {len(character_segments_with_lang)} segments with continuous workers")
        
        # Import streaming processor
        from engines.chatterbox.character_grouper import CharacterGrouper
        from engines.chatterbox.streaming_work_queue import StreamingWorkQueueProcessor
        
        # Build character groups for all languages
        character_groups_by_lang = {}
        for lang_code, lang_segments in language_groups.items():
            character_groups_by_lang[lang_code] = CharacterGrouper.group_by_character(lang_segments)
        
        # Initialize streaming processor  
        max_workers = inputs.get("batch_size", 4)
        streaming_processor = StreamingWorkQueueProcessor(max_workers=max_workers, tts_node=self)
        
        # Process all segments with streaming
        streaming_results = streaming_processor.process_streaming(
            language_groups=language_groups,
            character_groups_by_lang=character_groups_by_lang,
            voice_refs=voice_refs,
            inputs=inputs
        )
        
        # Convert streaming results to expected format
        audio_segments_with_order = []
        for original_idx, audio_tensor in streaming_results.items():
            audio_segments_with_order.append((original_idx, audio_tensor))
            
        return audio_segments_with_order

    def _process_languages_traditional(self, language_groups, voice_refs, inputs, character_segments_with_lang):
        """Process languages using traditional character-by-character method."""
        print(f"🎯 TRADITIONAL MODE: Processing {len(language_groups)} language groups sequentially")
        
        audio_segments_with_order = []
        current_loaded_language = None
        
        for original_idx, (char, segment_text, lang) in enumerate(character_segments_with_lang):
            # Load TTS model for this language if not already loaded
            from utils.models.language_mapper import get_model_for_language
            required_model = get_model_for_language("chatterbox", lang, "English")
            
            if current_loaded_language != required_model:
                print(f"🔄 Loading ChatterBox model '{required_model}' for {lang} language")
                self.load_tts_model(inputs["device"], required_model)
                current_loaded_language = required_model
            
            # Process each segment individually
            char_audio_prompt = voice_refs.get(char, voice_refs.get("narrator", "none"))
            
            segment_audio = self._generate_tts_with_pause_tags(
                segment_text, char_audio_prompt, inputs["exaggeration"],
                inputs["temperature"], inputs["cfg_weight"], lang,
                True, character=char, seed=inputs["seed"],
                enable_cache=inputs.get("enable_audio_cache", True),
                crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=""
            )
            
            audio_segments_with_order.append((original_idx, segment_audio))
            # Use the original better format with proper emoji based on character type
            if char == "narrator":
                print(f"🎤 Generating ChatterBox segment {original_idx+1}/{len(character_segments_with_lang)} for '{char}' (lang: {lang})")
            else:
                print(f"🎭 Generating ChatterBox segment {original_idx+1}/{len(character_segments_with_lang)} for '{char}' (lang: {lang})")
            
        return audio_segments_with_order

    def _process_single_segment_for_streaming(self, original_idx, character, segment_text, language, voice_path, inputs):
        """Process a single segment for the streaming processor using pre-loaded models."""
        # This method is called by the streaming worker
        try:
            # Get the pre-loaded model for this language (thread-safe)
            if hasattr(self, '_streaming_model_manager'):
                preloaded_model = self._streaming_model_manager.get_model_for_language(language)
                if preloaded_model:
                    # print(f"🧵 Worker using pre-loaded {language} model")  # Debug output - commented for cleaner console
                    # Temporarily switch to the pre-loaded model for this segment
                    original_model = self.tts_model
                    self.tts_model = preloaded_model
                    
                    try:
                        # Don't apply crash protection here - let _generate_tts_with_pause_tags handle it
                        # This preserves pause tag processing
                        segment_audio = self._generate_tts_with_pause_tags(
                            segment_text, voice_path, inputs.get("exaggeration", 0.5),
                            inputs.get("temperature", 0.8), inputs.get("cfg_weight", 0.5), language,
                            True, character=character, seed=inputs.get("seed", 42),
                            enable_cache=inputs.get("enable_audio_cache", True),
                            crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                            stable_audio_component=""
                        )
                        return segment_audio
                    finally:
                        # Restore original model
                        self.tts_model = original_model
            
            # Fallback to original method if no pre-loaded model
            print(f"⚠️ No pre-loaded model for {language}, using fallback")
            from utils.models.language_mapper import get_model_for_language
            required_model = get_model_for_language("chatterbox", language, "English")
            
            # Don't apply crash protection here - let _generate_tts_with_pause_tags handle it
            # This preserves pause tag processing
            segment_audio = self._generate_tts_with_pause_tags(
                segment_text, voice_path, inputs.get("exaggeration", 0.5),
                inputs.get("temperature", 0.8), inputs.get("cfg_weight", 0.5), language,
                True, character=character, seed=inputs.get("seed", 42),
                enable_cache=inputs.get("enable_audio_cache", True),
                crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=""
            )
            return segment_audio
            
        except Exception as e:
            print(f"❌ Streaming segment failed: {e}")
            # Return silence instead of crashing
            if hasattr(self, 'tts_model') and self.tts_model:
                sr = self.tts_model.sr
            else:
                sr = 22050  # Default sample rate
            return torch.zeros(1, int(sr * 1.0))  # 1 second of silence

    def _preload_language_models(self, language_codes, device):
        """Pre-load all required language models for streaming to prevent worker conflicts."""
        from engines.chatterbox.streaming_model_manager import StreamingModelManager
        
        # Create streaming model manager if not exists
        if not hasattr(self, '_streaming_model_manager'):
            self._streaming_model_manager = StreamingModelManager()
        
        # Pre-load models using the streaming model manager
        self._streaming_model_manager.preload_models(
            language_codes=list(language_codes),
            model_manager=self,  # Pass self as model_manager (has load_tts_model method)
            device=device
        )
        
        print(f"🚀 Pre-loading complete: {len(self._streaming_model_manager.preloaded_models)} models ready")

    def _pad_short_text_for_chatterbox(self, text: str, crash_protection_template: str, min_length: int = 15) -> str:
        """Apply crash protection padding to short texts to prevent GPU crashes."""
        if len(text.strip()) < min_length:
            protected_text = crash_protection_template.format(seg=text)
            print(f"🛡️ Crash protection applied: '{text}' -> '{protected_text}'")
            return protected_text
        return text