"""
ChatterBox Official 23-Lang Processor - Multilingual TTS processor
Enhanced Text-to-Speech processor using ChatterBox Official 23-Lang with multilingual support
"""

import torch
import numpy as np
import os
import gc
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import comfy.model_management as model_management

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
from utils.text.segment_parameters import apply_segment_parameters
import comfy.model_management as model_management

# Import the ChatterBox Official 23-Lang TTS engine
from engines.chatterbox_official_23lang.tts import ChatterboxOfficial23LangTTS



class ChatterboxOfficial23LangTTSNode(BaseTTSNode):
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
            from engines.chatterbox_official_23lang.language_models import get_supported_language_names
            available_languages = get_supported_language_names()
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
                "repetition_penalty": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Penalty for repeated tokens. Higher values reduce repetition in generated speech."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability threshold for token selection. Lower values allow more diverse tokens."
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling threshold. Controls the probability mass of tokens to consider."
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
        # Override to use ChatterBox Official 23-Lang engine instead of regular ChatterBox
        self.engine_type = "chatterbox_official_23lang"
        
        # Initialize model variable
        self.tts_model = None
        self.device = None
        self.current_language = None
        
        # Initialize ChatterBox Official 23-Lang model manager
        from engines.chatterbox_official_23lang import ChatterboxOfficial23LangTTS, ChatterboxOfficial23LangVC
        self.ChatterboxTTS = ChatterboxOfficial23LangTTS
        self.ChatterboxVC = ChatterboxOfficial23LangVC
    
    def load_tts_model(self, device: str, language: str, model_version: str = "v2"):
        """
        Override base method to load ChatterBox Official 23-Lang models using unified interface.
        """
        print(f"🌍 Loading ChatterBox Official 23-Lang model for {language} on {device}")

        # Use unified model interface for ComfyUI VRAM management
        from utils.models.unified_model_interface import load_tts_model

        try:
            # Load through unified interface which handles caching and VRAM management
            engine = load_tts_model(
                engine_name="chatterbox_official_23lang",
                model_name="Official 23-Lang",  # Always same model for ChatterBox 23-Lang
                language=language,
                device=device,
                model_version=model_version
            )

            print(f"✅ ChatterBox Official 23-Lang '{language}' loaded via unified interface")
            return engine
            
        except Exception as e:
            print(f"❌ Failed to load ChatterBox 23-Lang via unified interface: {e}")
            # Fallback to direct loading
            
            # Get the model directory for ChatterBox Official 23-Lang
            # This should load from: models/TTS/chatterbox_official_23lang/Official 23-Lang/
            import folder_paths
            
            models_dir = folder_paths.models_dir
        # Use the unified model directory for ChatterBox Official 23-Lang
        ckpt_dir = os.path.join(models_dir, "TTS", "chatterbox_official_23lang", "Official 23-Lang")
        
        print(f"📁 Loading from directory: {ckpt_dir}")
        
        # Auto-download model if it doesn't exist or if critical files are missing
        required_files = [
            "t3_23lang.safetensors",
            "s3gen.pt", 
            "ve.pt",
            "mtl_tokenizer.json"
        ]
        
        missing_files = []
        if not os.path.exists(ckpt_dir):
            missing_files = required_files
        else:
            for file in required_files:
                if not os.path.exists(os.path.join(ckpt_dir, file)):
                    # Check for alternative formats for s3gen and ve
                    if file.endswith('.pt') and file in ['s3gen.pt', 've.pt']:
                        # Check if safetensors version exists
                        safetensors_file = file.replace('.pt', '.safetensors')
                        if not os.path.exists(os.path.join(ckpt_dir, safetensors_file)):
                            missing_files.append(file)
                        # else: safetensors exists, so file is not missing
                    else:
                        missing_files.append(file)
        
        if missing_files:
            print(f"📥 ChatterBox Official 23-Lang missing files: {missing_files}")
            print("📥 Downloading missing files...")
            
            # Use unified downloader to download the model
            from utils.downloads.unified_downloader import unified_downloader
            
            # Try to download official .pt format first
            print("📥 Trying official .pt format...")
            success_dir = unified_downloader.download_chatterbox_model(
                repo_id="ResembleAI/chatterbox",
                model_name="Official 23-Lang",
                subdirectory=None,
                files=[
                    "t3_23lang.safetensors",
                    "s3gen.pt", 
                    "ve.pt",
                    "mtl_tokenizer.json",
                    "conds.pt"
                ]
            )
            
            # Fallback to safetensors if .pt download failed
            if not success_dir:
                print("⚠️ Official .pt download failed, trying safetensors fallback...")
                success_dir = unified_downloader.download_chatterbox_model(
                    repo_id="ResembleAI/chatterbox",
                    model_name="Official 23-Lang",
                    subdirectory=None,
                    files=[
                        "t3_23lang.safetensors",
                        "s3gen.safetensors", 
                        "ve.safetensors",
                        "mtl_tokenizer.json",
                        "conds.pt"
                    ]
                )
            
            if not success_dir:
                raise RuntimeError("Failed to download ChatterBox Official 23-Lang model (tried both .pt and safetensors)")
            
            print("✅ ChatterBox Official 23-Lang model downloaded successfully!")
        
        # Use the ChatterBox Official 23-Lang engine with correct parameters
        self.tts_model = ChatterboxOfficial23LangTTS.from_local(
            ckpt_dir=ckpt_dir,
            device=device,
            model_name="Official 23-Lang",
            model_version=model_version
        )
        
        self.device = device
        self.current_language = language
        
        # Return the model for smart_model_loader
        return self.tts_model
    
    def _language_name_to_code(self, language_input: str) -> str:
        """Convert language name or code to language code compatible with ChatterBox Official 23-Lang."""
        from utils.models.language_mapper import resolve_language_alias

        # First resolve using the centralized language mapper
        resolved_code = resolve_language_alias(language_input)
        
        # Map to ChatterBox Official 23-Lang supported codes
        # Import supported languages from our language models
        from engines.chatterbox_official_23lang.language_models import SUPPORTED_LANGUAGES
        
        # If it's already a supported language code, return it
        if resolved_code in SUPPORTED_LANGUAGES:
            return resolved_code
            
        # Handle special cases where character parser codes don't match Official 23-Lang
        code_mapping = {
            # Portuguese variations -> single pt
            "pt-br": "pt",  # Brazilian Portuguese -> Portuguese
            "pt-pt": "pt",  # European Portuguese -> Portuguese  
            "ptbr": "pt",
            "portuguese": "pt",
            # Greek
            "gr": "el",     # Greece -> Greek
            "greek": "el",
            # Any other unmapped codes
        }
        
        mapped_code = code_mapping.get(resolved_code, resolved_code)
        
        # If still not supported, default to English
        if mapped_code not in SUPPORTED_LANGUAGES:
            print(f"⚠️ Language tag '{language_input}' not recognized by ChatterBox Official 23-Lang. Falling back to English.")
            return "en"
            
        return mapped_code
    
    def generate_tts_audio(self, text: str, audio_prompt: str, exaggeration: float = 0.5, temperature: float = 0.8, cfg_weight: float = 0.5, repetition_penalty: float = 1.2, min_p: float = 0.05, top_p: float = 1.0, language_id: str = "en"):
        """
        Override base method to work directly with ChatterBox Official 23-Lang model.
        """
        if self.tts_model is None:
            raise RuntimeError("TTS model not loaded. Call load_tts_model() first.")

        # CRITICAL FIX: Reload model to correct device if it was offloaded
        # When processors are cached, the model reference persists but may be on wrong device
        # Use the unified model interface to properly reload through wrapper system
        if hasattr(self.tts_model, 'to') and hasattr(self.tts_model, 'device'):
            target_device = self.device if hasattr(self, 'device') else self.tts_model.device

            # Resolve "auto" to actual device
            if target_device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"

            # Check current device of model components
            if hasattr(self.tts_model.t3, 'parameters'):
                try:
                    current_device = str(next(self.tts_model.t3.parameters()).device)
                    target_device_str = str(target_device)

                    # Reload through unified interface if device mismatch
                    # This ensures ComfyUI's model management stays in sync
                    if current_device != target_device_str:
                        # print(f"🔄 Reloading ChatterBox model from {current_device} to {target_device_str} via wrapper")

                        # Find and call wrapper's model_load() instead of direct .to()
                        try:
                            from utils.models.comfyui_model_wrapper.model_manager import tts_model_manager
                            for cache_key, wrapper in tts_model_manager._model_cache.items():
                                if hasattr(wrapper, 'model') and wrapper.model is self.tts_model:
                                    wrapper.model_load(target_device)
                                    # print(f"✅ Reloaded model via wrapper - ComfyUI management stays in sync")
                                    break
                            else:
                                # Fallback: direct .to() if wrapper not found
                                print(f"⚠️ Wrapper not found, using direct .to() - 'Clear VRAM' may not work")
                                self.tts_model.to(target_device)
                        except Exception as e:
                            # Fallback to direct .to()
                            print(f"⚠️ Wrapper reload failed ({e}), using direct .to()")
                            self.tts_model.to(target_device)
                except StopIteration:
                    pass  # Model has no parameters

        # Use torch.no_grad() to ensure no gradients are tracked during inference
        with torch.no_grad():
            # Debug: Show the language_id being passed to the model
            print(f"🌍 ChatterBox Official 23-Lang TTS: language_id='{language_id}' for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # ChatterBox generate method with correct parameters including language_id
            audio = self.tts_model.generate(
                text,
                language_id,
                audio_prompt_path=audio_prompt if audio_prompt else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p
            )
            return audio
    
    def _pad_short_text_for_chatterbox(self, text: str, crash_protection_template: str = "hmm ,, {seg} hmm ,,", min_length: int = 15) -> str:
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
            crash_protection_template: Custom template with {seg} placeholder for original text
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
            if not crash_protection_template.strip():
                return text
            # Replace {seg} placeholder with original text
            protected_text = crash_protection_template.replace("{seg}", stripped_text)
            print(f"🛡️ Crash protection applied: '{text}' -> '{protected_text}'")
            return protected_text
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
            # padded_text = self._pad_short_text_for_chatterbox(text)  # DISABLED FOR TESTING
            padded_text = text  # Direct text without crash protection
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
            language_code = self._language_name_to_code(language)
            return self.generate_tts_audio(text, audio_prompt, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, language_code)
        
        # Predict and skip problematic text before it crashes
        # The text passed here is already processed/padded, so check it directly
        is_problematic, reason = self._is_problematic_text(text, is_already_padded=True)
        if is_problematic:
            print(f"🚨 SKIPPING PROBLEMATIC SEGMENT: '{text[:50]}...' - Reason: {reason}")
            print(f"🛡️ Generating silence to prevent ChatterBox CUDA crash and avoid ComfyUI reboot")
            # Return silence instead of attempting generation
            silence_duration = max(1.0, len(text) * 0.05)  # Rough estimate
            silence_samples = int(silence_duration * (self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000))
            return torch.zeros(1, silence_samples)
        
        # If prediction says it's safe, try generation with fallback
        try:
            language_code = self._language_name_to_code(language)
            return self.generate_tts_audio(text, audio_prompt, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, language_code)
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
                silence_samples = int(silence_duration * (self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000))
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
            # Convert v2 special tags (AFTER character parsing, BEFORE TTS engine)
            if hasattr(self.tts_model, 'model_version') and self.tts_model.model_version == "v2":
                from utils.text.chatterbox_v2_special_tags import convert_v2_special_tags
                segment_text = convert_v2_special_tags(segment_text)

            # Apply padding for crash protection
            # processed_text = self._pad_short_text_for_chatterbox(segment_text, inputs["crash_protection_template"])  # DISABLED FOR TESTING
            processed_text = segment_text  # Direct text without crash protection
            
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
            for char_name in all_characters:
                audio_path, _ = character_mapping.get(char_name, (None, None))
                voice_refs[char_name] = audio_path if audio_path else main_audio_prompt
        
        # Generate audio using pause tag processor
        def tts_generate_func(text_content: str) -> torch.Tensor:
            """TTS generation function for pause tag processor"""
            if has_character_switching and ('[' in text_content and ']' in text_content):
                # Handle character switching within this segment
                char_segments = parse_character_text(text_content)
                segment_audio_parts = []
                
                for char_name, segment_text in char_segments:
                    audio_prompt = voice_refs.get(char_name, main_audio_prompt)
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
                           silence_ms: int, text_length: int, original_text: str = "",
                           text_chunks: List[str] = None, return_info: bool = False):
        """Combine audio segments using modular combination utility."""
        if len(audio_segments) == 1:
            if return_info:
                sr = self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000
                chunk_info = {
                    "method_used": "none",
                    "total_chunks": 1,
                    "chunk_timings": [{"start": 0.0, "end": audio_segments[0].size(-1) / sr, 
                                     "text": text_chunks[0] if text_chunks else ""}],
                    "auto_selected": False
                }
                return audio_segments[0], chunk_info
            return audio_segments[0]
        
        print(f"🔗 Combining {len(audio_segments)} ChatterBox chunks using '{method}' method")
        
        # Use modular chunk combiner
        from utils.audio.chunk_combiner import ChunkCombiner
        sr = self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000
        
        result = ChunkCombiner.combine_chunks(
            audio_segments=audio_segments,
            method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=sr,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
            return_info=return_info
        )
        
        if return_info:
            return result  # (combined_audio, chunk_info)
        else:
            return result  # combined_audio
    
    def _generate_stable_audio_component(self, reference_audio, audio_prompt_path: str) -> str:
        """Generate stable identifier for audio prompt to prevent cache invalidation from temp file paths."""
        # Use robust import system (fix for issue #12)
        from utils.robust_import import robust_from_import
        attrs = robust_from_import('utils.audio.audio_hash', ['generate_stable_audio_component'])
        generate_stable_audio_component = attrs['generate_stable_audio_component']
        return generate_stable_audio_component(reference_audio, audio_prompt_path)


    def _generate_tts_with_pause_tags(self, text: str, audio_prompt, exaggeration: float,
                                    temperature: float, cfg_weight: float, repetition_penalty: float = 1.2,
                                    min_p: float = 0.05, top_p: float = 1.0, language: str = "English",
                                    enable_pause_tags: bool = True, character: str = "narrator",
                                    seed: int = 0, enable_cache: bool = True,
                                    crash_protection_template: str = "hmm ,, {seg} hmm ,,",
                                    stable_audio_component: str = None, model_version: str = "v1") -> torch.Tensor:
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

        # Convert v2 special tags AFTER pause processing, BEFORE TTS generation
        if model_version == "v2":
            from utils.text.chatterbox_v2_special_tags import convert_v2_special_tags
            processed_text = convert_v2_special_tags(processed_text)
            # Also convert in pause segments if they exist
            if pause_segments is not None:
                pause_segments = [
                    (seg_type, convert_v2_special_tags(content) if seg_type == 'text' else content)
                    for seg_type, content in pause_segments
                ]

        # Debug pause tag processing in streaming
        if pause_segments is not None:
            print(f"🏷️ PAUSE TAGS: Found in '{text[:50]}...' -> {len(pause_segments)} segments")
        
        if pause_segments is None:
            # No pause tags, use regular generation with caching
            if enable_cache:
                # Use stable audio component for cache key
                audio_component = stable_audio_component if stable_audio_component else ""
                
                # Apply crash protection first for consistency
                # protected_text = self._pad_short_text_for_chatterbox(processed_text, crash_protection_template)  # DISABLED FOR TESTING
                protected_text = processed_text  # Direct text without crash protection
                
                # Show final text going into the TTS model
                print(f"🔤 Final text to ChatterBox TTS model ({character}): '{protected_text}'")
                
                # Use centralized cache system
                from utils.audio.cache import create_cache_function
                cache_fn = create_cache_function(
                    "chatterbox_official_23lang",
                    character=character,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    audio_component=audio_component,
                    model_source=f"chatterbox_{language.lower()}",
                    model_version=model_version,
                    device=self.device,
                    language=language,
                    # Add ChatterBox Official 23-Lang specific parameters to cache key
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p
                )
                
                # Try cache first
                cached_audio = cache_fn(protected_text)
                if cached_audio is not None:
                    print(f"💾 CACHE HIT for {character}: '{processed_text[:30]}...'")
                    return cached_audio
                
                # Generate and cache
                language_code = self._language_name_to_code(language)
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, language_code)
                # Clone tensor to avoid autograd issues in streaming mode
                audio_clone = audio.detach().clone() if audio.requires_grad else audio
                cache_fn(protected_text, audio_result=audio_clone)
                return audio_clone
            else:
                # protected_text = self._pad_short_text_for_chatterbox(processed_text, crash_protection_template)  # DISABLED FOR TESTING
                protected_text = processed_text  # Direct text without crash protection
                # Show final text going into the TTS model
                print(f"🔤 Final text to ChatterBox TTS model ({character}): '{protected_text}'")
                language_code = self._language_name_to_code(language)
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, language_code)
                # Clone tensor to avoid autograd issues in streaming mode
                return audio.detach().clone() if audio.requires_grad else audio
        
        # Generate audio with pause tags, caching individual text segments
        def tts_generate_func(text_content: str) -> torch.Tensor:
            """TTS generation function for pause tag processor with caching"""
            if enable_cache:
                # Use stable audio component for cache key
                audio_component = stable_audio_component if stable_audio_component else ""
                
                # Apply crash protection first for consistency
                # protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)  # DISABLED FOR TESTING
                protected_text = text_content  # Direct text without crash protection
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                # Show final text going into the TTS model
                print(f"🔤 Final text to ChatterBox TTS model ({character}, pause segment): '{protected_text}'")
                
                # Use centralized cache system
                from utils.audio.cache import create_cache_function
                cache_fn = create_cache_function(
                    "chatterbox_official_23lang",
                    character=character,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    seed=seed,
                    audio_component=audio_component,
                    model_source=f"chatterbox_{language.lower()}",
                    model_version=model_version,
                    device=self.device,
                    language=language,
                    # Add ChatterBox Official 23-Lang specific parameters to cache key
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p
                )
                
                # Try cache first
                cached_audio = cache_fn(protected_text)
                if cached_audio is not None:
                    print(f"💾 CACHE HIT for {character}: '{text_content[:30]}...'")
                    return cached_audio
                
                # Generate and cache
                language_code = self._language_name_to_code(language)
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, language_code)
                # Clone tensor to avoid autograd issues in streaming mode
                audio_clone = audio.detach().clone() if audio.requires_grad else audio
                cache_fn(protected_text, audio_result=audio_clone)
                return audio_clone
            else:
                # Apply crash protection
                # protected_text = self._pad_short_text_for_chatterbox(text_content, crash_protection_template)  # DISABLED FOR TESTING
                protected_text = text_content  # Direct text without crash protection
                if len(text_content.strip()) < 21:
                    print(f"🔍 DEBUG: Pause segment original: '{text_content}' → Protected: '{protected_text}' (len: {len(protected_text)})")
                
                # Show final text going into the TTS model
                print(f"🔤 Final text to ChatterBox TTS model ({character}, pause segment, no cache): '{protected_text}'")
                
                language_code = self._language_name_to_code(language)
                audio = self.generate_tts_audio(protected_text, audio_prompt, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, language_code)
                # Clone tensor to avoid autograd issues in streaming mode
                return audio.detach().clone() if audio.requires_grad else audio
        
        return PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, tts_generate_func, self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000
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
            inputs["temperature"], inputs["cfg_weight"], inputs["repetition_penalty"],
            inputs["min_p"], inputs["top_p"], inputs["language"],
            True, character=inputs["character"], seed=inputs["seed"],
            enable_cache=inputs.get("enable_audio_cache", True),
            crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
            stable_audio_component=stable_audio_component,
            model_version=inputs.get("model_version", "v1")
        )

    def generate_speech(self, text, language, device, model_version="v2", exaggeration=0.5, temperature=0.8, cfg_weight=0.5,
                       repetition_penalty=2.0, min_p=0.05, top_p=1.0, seed=0,
                       reference_audio=None, audio_prompt_path="",
                       enable_chunking=True, max_chars_per_chunk=400,
                       chunk_combination_method="auto", silence_between_chunks_ms=100,
                       enable_audio_cache=True, batch_size=4, character="narrator"):
        
        def _process():
            # Import PauseTagProcessor at the top to avoid scoping issues
            from utils.text.pause_processor import PauseTagProcessor
            
            # Capture batch_size from outer scope to avoid UnboundLocalError
            current_batch_size = batch_size
            
            # Validate inputs
            inputs = self.validate_inputs(
                text=text, language=language, device=device, model_version=model_version,
                exaggeration=exaggeration,
                temperature=temperature, cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty, min_p=min_p, top_p=top_p,
                seed=seed, reference_audio=reference_audio, audio_prompt_path=audio_prompt_path,
                enable_chunking=enable_chunking, max_chars_per_chunk=max_chars_per_chunk,
                chunk_combination_method=chunk_combination_method,
                silence_between_chunks_ms=silence_between_chunks_ms,
                enable_audio_cache=enable_audio_cache,
                batch_size=current_batch_size,
                character=character  # Add character to inputs
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
            
            # Parse character segments from original text for all modes (with Italian prefix automatically applied)
            # Parse using CharacterSegment objects to get parameters
            character_segment_objects = character_parser.parse_text_segments(inputs["text"])

            # Create backward-compatible segments from CharacterSegment objects (Italian prefix already applied in parser)
            character_segments_with_lang = [(seg.character, seg.text, seg.language, seg.parameters if seg.parameters else {}) for seg in character_segment_objects]
            character_segments_with_lang_and_explicit = [(seg.character, seg.text, seg.language, seg.explicit_language) for seg in character_segment_objects]
            
            # Check if we have pause tags, character switching, or language switching
            has_pause_tags = pause_segments is not None
            characters = list(set(char for char, _, _, _ in character_segments_with_lang))
            languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))
            has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
            has_multiple_languages = len(languages) > 1

            # Create backward-compatible character segments for existing logic
            character_segments = [(char, segment_text) for char, segment_text, _, _ in character_segments_with_lang]
            
            if has_multiple_characters or has_multiple_languages:
                # CHARACTER AND/OR LANGUAGE SWITCHING MODE
                if has_multiple_languages:
                    print(f"🌍 ChatterBox: Language switching mode - found languages: {', '.join(languages)}")
                if has_multiple_characters:
                    print(f"🎭 ChatterBox: Character switching mode - found characters: {', '.join(characters)}")
                
                
                # Get character voice mapping (ChatterBox doesn't need reference text)
                character_mapping = get_character_mapping(characters, engine_type="chatterbox")
                
                # Build voice references with fallback to main voice
                # CRITICAL FIX: Explicitly map narrator to selected voice (matches SRT node logic)
                voice_refs = {'narrator': main_audio_prompt or None}
                character_voices = []
                main_voices = []
                
                for char_name in characters:
                    # CRITICAL FIX: Skip narrator - it should use selected input/dropdown voice, not character voice files
                    if char_name == 'narrator':
                        continue
                        
                    audio_path, _ = character_mapping.get(char_name, (None, None))
                    if audio_path:
                        voice_refs[char_name] = audio_path
                        character_voices.append(char_name)
                        
                        # CRITICAL FIX: Also map resolved character name to same audio path
                        # This ensures streaming workers can find voices using resolved names
                        from utils.voice.discovery import voice_discovery
                        resolved_name = voice_discovery.resolve_character_alias(char_name)
                        if resolved_name != char_name:
                            voice_refs[resolved_name] = audio_path
                            
                    else:
                        voice_refs[char_name] = main_audio_prompt
                        main_voices.append(char_name)
                
                # Consolidated voice summary logging
                voice_summary = []
                if character_voices:
                    voice_summary.append(f"character voices: {', '.join(character_voices)}")
                if main_voices:
                    voice_summary.append(f"main voice: {', '.join(main_voices)}")
                
                if voice_summary:
                    print(f"🎭 Voice mapping - {' | '.join(voice_summary)}")
                
                # Map language codes to ChatterBox model names
                
                # Preprocess pause tags in character segments before streaming
                expanded_segments_with_lang = []
                pause_info = {}  # Track pause information for reconstruction
                segment_mapping = {}  # Map streaming indices to original indices
                streaming_idx = 0

                for original_idx, seg_obj in enumerate(character_segment_objects):
                    char = seg_obj.character
                    segment_text = seg_obj.text
                    lang = seg_obj.language
                    segment_params = seg_obj.parameters if seg_obj.parameters else {}

                    from utils.text.pause_processor import PauseTagProcessor
                    processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(segment_text, True)

                    if pause_segments is not None:
                        # Segment has pause tags - expand into multiple sub-segments
                        print(f"⏸️ Expanding segment {original_idx} with pause tags: {len(pause_segments)} parts")
                        sub_idx = 0
                        for segment_type, content in pause_segments:
                            if segment_type == 'text' and content.strip():
                                # Text segment - add to streaming queue with parameters
                                expanded_segments_with_lang.append((streaming_idx, char, content, lang, segment_params))
                                segment_mapping[streaming_idx] = f"{original_idx}_{sub_idx}"
                                streaming_idx += 1
                                sub_idx += 1
                            elif segment_type == 'pause':
                                # Store pause info for later reconstruction
                                pause_key = f"{original_idx}_{sub_idx}"
                                pause_info[pause_key] = content  # pause duration
                                sub_idx += 1
                    else:
                        # No pause tags - add as single segment with parameters
                        expanded_segments_with_lang.append((streaming_idx, char, segment_text, lang, segment_params))
                        segment_mapping[streaming_idx] = original_idx
                        streaming_idx += 1
                
                # Group expanded segments by language with original order tracking
                language_groups = {}
                for idx, char, segment_text, lang, segment_params in expanded_segments_with_lang:
                    if lang not in language_groups:
                        language_groups[lang] = []
                    language_groups[lang].append((idx, char, segment_text, lang, segment_params))  # Include parameters
                
                # Generate audio for each language group, tracking original positions
                audio_segments_with_order = []  # Will store (original_index, audio_tensor)
                total_segments = len(expanded_segments_with_lang)
                
                # Choose processing method based on batch_size: 0-1 = sequential, 2+ = streaming
                current_batch_size = inputs.get("batch_size", 1)
                use_streaming = (current_batch_size > 1)  # Let user decide: batch_size > 1 = streaming, regardless of segment count
                
                if use_streaming:
                    # Pre-load ALL language models for streaming efficiency (prevents worker conflicts)
                    print(f"🚀 STREAMING: Pre-loading models for {len(language_groups)} languages")
                    self._preload_language_models(language_groups.keys(), inputs["device"])
                    
                    audio_segments_with_order = self._process_languages_streaming(
                        language_groups, voice_refs, inputs, expanded_segments_with_lang, pause_info, segment_mapping
                    )
                else:
                    audio_segments_with_order = self._process_languages_traditional(
                        language_groups, voice_refs, inputs, character_segments_with_lang
                    )
                
                # Continue to sorting and combining...
                audio_segments_with_order.sort(key=lambda x: x[0])  # Sort by original index
                audio_segments = [audio for _, audio in audio_segments_with_order]  # Extract audio tensors
                
                # Create processed text for timing display (character tags removed, Italian prefixes applied)
                processed_text_segments = [segment_text for _, segment_text, _, _ in character_segments_with_lang]
                processed_text = ' '.join(processed_text_segments)
                
                # Combine all character segments with timing info
                wav, chunk_info = self.combine_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], len(processed_text),
                    original_text=processed_text, text_chunks=None, return_info=True
                )
                
                # Generate info - handle case where streaming was used and tts_model is None
                sample_rate = self.tts_model.sr if self.tts_model else 24000  # ChatterBox default
                total_duration = wav.size(-1) / sample_rate
                model_source = f"chatterbox_{language.lower()}"
                
                language_info = ""
                if has_multiple_languages:
                    language_info = f" across {len(languages)} languages ({', '.join(languages)})"
                
                base_info = f"Generated {total_duration:.1f}s audio from {len(character_segments)} segments using {len(characters)} characters{language_info} ({model_source} models)"
                
                # Add chunk timing info if available
                from utils.audio.chunk_timing import ChunkTimingHelper
                info = ChunkTimingHelper.enhance_generation_info(base_info, chunk_info)
                
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
                            # Check centralized cache system
                            from utils.audio.cache import create_cache_function
                            cache_fn = create_cache_function(
                                "chatterbox_official_23lang",
                                character=inputs["character"],
                                exaggeration=inputs["exaggeration"],
                                temperature=inputs["temperature"],
                                cfg_weight=inputs["cfg_weight"],
                                seed=inputs["seed"],
                                audio_component=stable_audio_component,
                                model_source=f"chatterbox_{language.lower()}",
                                model_version=inputs.get("model_version", "v1"),
                                device=inputs["device"],
                                language=inputs["language"]
                            )
                            cached_data = cache_fn(f"narrator:{cache_text}")
                            cached_data = None if cached_data is None else (cached_data, 0.0)  # Convert to tuple format
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
                                # Check centralized cache system
                                from utils.audio.cache import create_cache_function
                                cache_fn = create_cache_function(
                                    "chatterbox_official_23lang",
                                    character=inputs["character"],
                                    exaggeration=inputs["exaggeration"],
                                    temperature=inputs["temperature"],
                                    cfg_weight=inputs["cfg_weight"],
                                    seed=inputs["seed"],
                                    audio_component=stable_audio_component,
                                    model_source=f"chatterbox_{language.lower()}",
                                    model_version=inputs.get("model_version", "v1"),
                                    device=inputs["device"],
                                    language=inputs["language"]
                                )
                                cached_data = cache_fn(f"narrator:{cache_text}")
                                cached_data = None if cached_data is None else (cached_data, 0.0)  # Convert to tuple format
                                if not cached_data:
                                    single_content_cached = False
                                    break
                            if not single_content_cached:
                                break
                
                # Only load model if we need to generate something
                if not single_content_cached:
                    # Use unified model interface for ComfyUI VRAM management
                    if not hasattr(self, 'tts_model') or self.tts_model is None:
                        self.tts_model = self.load_tts_model(inputs["device"], inputs["language"], inputs.get("model_version", "v2"))
                        self.device = inputs["device"]  # Update device tracking
                else:
                    print(f"💾 All single character content cached - skipping model loading")
                
                if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                    # Process single chunk with caching support
                    # BUGFIX: Clean character tags from text even in single character mode
                    clean_text = character_parser.remove_character_tags(inputs["text"])
                    wav = self._generate_tts_with_pause_tags(
                        clean_text, main_audio_prompt, inputs["exaggeration"],
                        inputs["temperature"], inputs["cfg_weight"], inputs["repetition_penalty"],
                        inputs["min_p"], inputs["top_p"], inputs["language"],
                        True, character=inputs["character"], seed=inputs["seed"],
                        enable_cache=inputs.get("enable_audio_cache", True),
                        crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                        stable_audio_component=stable_audio_component,
                        model_version=inputs.get("model_version", "v1")
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
                            inputs["temperature"], inputs["cfg_weight"], inputs["repetition_penalty"],
                            inputs["min_p"], inputs["top_p"], inputs["language"],
                            True, character=inputs["character"], seed=inputs["seed"],
                            enable_cache=inputs.get("enable_audio_cache", True),
                            crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                            stable_audio_component=stable_audio_component,
                            model_version=inputs.get("model_version", "v1")
                        )
                        audio_segments.append(chunk_audio)
                    
                    # Create processed text for timing display (character tags removed, Italian prefixes applied)
                    processed_text_segments = [segment_text for _, segment_text, _ in character_segments_with_lang]
                    processed_text = ' '.join(processed_text_segments)
                    
                    # Combine audio segments with timing info
                    wav, chunk_info = self.combine_audio_chunks(
                        audio_segments, inputs["chunk_combination_method"], 
                        inputs["silence_between_chunks_ms"], len(processed_text),
                        original_text=processed_text, text_chunks=None, return_info=True
                    )
                    
                    # Generate info (UNCHANGED)
                    total_duration = wav.size(-1) / self.tts_model.sr
                    avg_chunk_size = text_length // len(chunks)
                    model_source = f"chatterbox_{language.lower()}"
                    base_info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, {model_source} models)"
                    
                    # Add chunk timing info if available
                    from utils.audio.chunk_timing import ChunkTimingHelper
                    info = ChunkTimingHelper.enhance_generation_info(base_info, chunk_info)
            
            # Return audio in ComfyUI format
            sr = self.tts_model.sr if hasattr(self, 'tts_model') and self.tts_model else 24000
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
                inputs["temperature"], inputs["cfg_weight"], inputs["repetition_penalty"],
                inputs["min_p"], inputs["top_p"], required_language,
                True, character=inputs["character"], seed=inputs.get("seed", 42),
                enable_cache=inputs.get("enable_audio_cache", True),
                crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=stable_audio_component,
                model_version=inputs.get("model_version", "v1")
            )
            segment_audio_chunks.append(chunk_audio)
        
        # Combine chunks for this segment and store with original order
        if segment_audio_chunks:
            if len(segment_audio_chunks) == 1:
                segment_audio = segment_audio_chunks[0]
            else:
                segment_audio = torch.cat(segment_audio_chunks, dim=-1)
            audio_segments_with_order.append((original_idx, segment_audio))

    def _process_languages_streaming(self, language_groups, voice_refs, inputs, expanded_segments_with_lang, pause_info=None, segment_mapping=None):
        """Process all languages using universal streaming system."""
        print(f"🌊 STREAMING MODE: Processing all {len(expanded_segments_with_lang)} segments with universal streaming")
        
        # Import universal streaming components
        from utils.streaming import StreamingCoordinator, StreamingConfig
        from engines.adapters.chatterbox_streaming_adapter import ChatterBoxStreamingAdapter
        
        # Convert expanded_segments_with_lang to indexed format for streaming
        # expanded_segments_with_lang is (idx, char, text, lang, params)
        indexed_segments = [(idx, char, text, lang, segment_params) for idx, char, text, lang, segment_params in expanded_segments_with_lang]
        
        # Convert to universal streaming segments
        segments = StreamingCoordinator.convert_node_data_to_segments(
            node_type='tts',
            data=indexed_segments,  # List of (idx, char, text, lang) tuples
            voice_refs=voice_refs
        )
        
        # Create streaming configuration
        config = StreamingConfig(
            batch_size=inputs.get("batch_size", 4),
            enable_model_preloading=True,
            fallback_to_traditional=True,
            streaming_threshold=1,
            engine_config={
                'device': inputs.get('device', 'auto'),
                'enable_audio_cache': inputs.get('enable_audio_cache', True)
            }
        )
        
        # Create ChatterBox streaming adapter
        adapter = ChatterBoxStreamingAdapter(self)
        
        # Process with universal streaming coordinator
        results, metrics, success = StreamingCoordinator.process(
            segments=segments,
            adapter=adapter,
            config=config,
            **inputs
        )
        
        # Convert results to expected format (maintain compatibility)
        audio_segments_with_order = []
        for original_idx in sorted(results.keys()):
            audio_segments_with_order.append((original_idx, results[original_idx]))
        
        # Print performance summary
        if success:
            summary = metrics.get_summary()
            print(f"✅ Streaming complete: {summary['completed_segments']}/{summary['total_segments']} segments, "
                  f"{summary['throughput']:.2f} segments/sec")
        
        # Reconstruct pauses if we have pause information
        if pause_info and success and segment_mapping:
            print(f"⏸️ Reconstructing {len(pause_info)} pauses in streaming results")
            audio_segments_with_order = self._reconstruct_pauses_in_streaming_results(
                audio_segments_with_order, pause_info, segment_mapping
            )
            
        return audio_segments_with_order

    def _reconstruct_pauses_in_streaming_results(self, audio_segments_with_order, pause_info, segment_mapping):
        """Reconstruct pause segments in streaming results."""
        from utils.audio.processing import AudioProcessingUtils
        
        # Create a mapping of streaming indices to their original segment identifiers
        # audio_segments_with_order contains (streaming_idx, audio)
        # segment_mapping maps streaming_idx -> original_segment_id
        
        # Group segments by original segment (before pause expansion)
        segment_groups = {}
        for streaming_idx, audio in audio_segments_with_order:
            original_segment_id = segment_mapping.get(streaming_idx, streaming_idx)
            
            if '_' in str(original_segment_id):
                # Expanded segment like "0_0", "0_1"
                original_idx = str(original_segment_id).split('_')[0]
            else:
                # Regular segment
                original_idx = str(original_segment_id)
            
            if original_idx not in segment_groups:
                segment_groups[original_idx] = []
            segment_groups[original_idx].append((original_segment_id, audio))
        
        # Reconstruct each original segment with pauses
        reconstructed_segments = []
        for original_idx in sorted(segment_groups.keys(), key=int):
            segment_parts = segment_groups[original_idx]
            
            if len(segment_parts) == 1 and '_' not in str(segment_parts[0][0]):
                # Single segment without pause expansion - no reconstruction needed
                reconstructed_segments.append((int(original_idx), segment_parts[0][1]))
            else:
                # Reconstruct with pauses
                combined_audio_parts = []
                
                # Sort parts by sub-index (parts are identified as "0_0", "0_1", etc.)
                segment_parts.sort(key=lambda x: int(str(x[0]).split('_')[1]) if '_' in str(x[0]) else 0)
                
                for i, (part_id, audio) in enumerate(segment_parts):
                    combined_audio_parts.append(audio)
                    
                    # Check if there's a pause after this part
                    # For expanded segments like "0_0", the next pause would be "0_1" 
                    if '_' in str(part_id):
                        current_sub_idx = int(str(part_id).split('_')[1])
                        pause_key = f"{original_idx}_{current_sub_idx + 1}"
                    else:
                        pause_key = f"{original_idx}_1"
                    
                    if pause_key in pause_info:
                        pause_duration = pause_info[pause_key]
                        sample_rate = 24000  # ChatterBox default
                        silence = AudioProcessingUtils.create_silence(pause_duration, sample_rate)
                        combined_audio_parts.append(silence)
                        print(f"⏸️ Added {pause_duration}s pause after segment {part_id}")
                
                # Combine all parts
                combined_audio = AudioProcessingUtils.concatenate_audio_segments(combined_audio_parts, "simple")
                reconstructed_segments.append((int(original_idx), combined_audio))
        
        return reconstructed_segments

    def _process_languages_traditional(self, language_groups, voice_refs, inputs, character_segments_with_lang):
        """Process languages using traditional character-by-character method."""
        print(f"🎯 TRADITIONAL MODE: Processing {len(language_groups)} language groups sequentially")
        
        audio_segments_with_order = []
        
        # For ChatterBox Official 23-Lang, we only need to load the model once
        # It's a multilingual model that handles all languages with the same model
        if not hasattr(self, 'tts_model') or self.tts_model is None:
            # Use unified model interface for ComfyUI VRAM management
            self.tts_model = self.load_tts_model(inputs["device"], inputs["language"], inputs.get("model_version", "v2"))
            self.device = inputs["device"]  # Update device tracking
        
        for original_idx, (char, segment_text, lang, segment_params) in enumerate(character_segments_with_lang):
            # For Official 23-Lang, we don't need to reload model for different languages
            # Just use the same model with different language_id parameter

            # Apply segment parameters if provided
            current_params = inputs.copy()
            if segment_params:
                segment_config = apply_segment_parameters(current_params, segment_params, "chatterbox_official_23lang")
                current_params.update(segment_config)
                if segment_params:
                    print(f"  📊 Segment {original_idx + 1}: Character '{char}' with params {segment_params}")

            # Process each segment individually
            char_audio_prompt = voice_refs.get(char, voice_refs.get("narrator", "none"))

            # Generate stable audio component for cache consistency
            stable_audio_component = self._generate_stable_audio_component(inputs.get("reference_audio"), char_audio_prompt)

            segment_audio = self._generate_tts_with_pause_tags(
                segment_text, char_audio_prompt, current_params["exaggeration"],
                current_params["temperature"], current_params["cfg_weight"], current_params["repetition_penalty"],
                current_params["min_p"], current_params["top_p"], lang,
                True, character=char, seed=current_params["seed"],
                enable_cache=current_params.get("enable_audio_cache", True),
                crash_protection_template=current_params.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=stable_audio_component,
                model_version=current_params.get("model_version", "v1")
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
            # Get the stateless wrapper for thread safety (same as SRT node)
            if hasattr(self, '_streaming_model_manager'):
                stateless_model = self._streaming_model_manager.get_stateless_model_for_language(language)
                if stateless_model:
                    # Process text for generation
                    # processed_text = self._pad_short_text_for_chatterbox(segment_text, inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"))  # DISABLED FOR TESTING
                    processed_text = segment_text  # Direct text without crash protection
                    
                    # Add caching logic like SRT streaming does
                    enable_cache = inputs.get("enable_audio_cache", True)
                    cached_audio = None
                    
                    # Try cache first
                    if enable_cache:
                        from utils.audio.cache import create_cache_function
                        cache_fn = create_cache_function(
                            "chatterbox_official_23lang",
                            character=inputs["character"],
                            exaggeration=inputs.get("exaggeration", 0.5),
                            temperature=inputs.get("temperature", 0.8),
                            cfg_weight=inputs.get("cfg_weight", 0.5),
                            seed=inputs.get("seed", 42),
                            audio_component=self._generate_stable_audio_component(inputs.get("reference_audio"), voice_path),
                            model_source="streaming_stateless",
                            model_version=inputs.get("model_version", "v1"),
                            device="auto",
                            language=language
                        )
                        cached_audio = cache_fn(processed_text)
                    
                    if cached_audio is not None:
                        print(f"💾 CACHE HIT for segment: '{processed_text[:20]}...'")
                        return cached_audio
                    else:
                        print(f"🔒 Using stateless wrapper for thread-safe generation")
                        
                        # Generate using stateless wrapper (thread-safe)
                        segment_audio = stateless_model.generate_stateless(
                            text=processed_text,
                            audio_prompt_path=voice_path if voice_path != "none" else None,
                            exaggeration=inputs.get("exaggeration", 0.5),
                            temperature=inputs.get("temperature", 0.8),
                            cfg_weight=inputs.get("cfg_weight", 0.5),
                            seed=inputs.get("seed", 42)
                        )
                        
                        # Cache the result
                        if enable_cache:
                            cache_fn(processed_text, audio_result=segment_audio)
                            print(f"💾 CACHED segment: '{processed_text[:20]}...'")
                        
                        return segment_audio
            
            # Fallback to old method if no streaming model manager
            print(f"⚠️ No stateless wrapper available, using fallback method")
            
            # Generate stable audio component for cache consistency
            stable_audio_component = self._generate_stable_audio_component(inputs.get("reference_audio"), voice_path)
            
            # Directly call the pause-aware generation method
            segment_audio = self._generate_tts_with_pause_tags(
                segment_text, voice_path, inputs.get("exaggeration", 0.5),
                inputs.get("temperature", 0.8), inputs.get("cfg_weight", 0.5),
                inputs.get("repetition_penalty", 1.2), inputs.get("min_p", 0.05), inputs.get("top_p", 1.0),
                language, True, character=inputs["character"], seed=inputs.get("seed", 42),
                enable_cache=inputs.get("enable_audio_cache", True),
                crash_protection_template=inputs.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                stable_audio_component=stable_audio_component,
                model_version=inputs.get("model_version", "v1")
            )
            return segment_audio
            
        except Exception as e:
            print(f"❌ Streaming segment failed: {e}")
            # Return silence instead of crashing
            if hasattr(self, 'tts_model') and self.tts_model:
                sr = self.tts_model.sr
            else:
                sr = 24000  # Default sample rate
            return torch.zeros(1, int(sr * 1.0))  # 1 second of silence

    def _preload_language_models(self, language_codes, device):
        """Pre-load all required language models for streaming to prevent worker conflicts."""
        from engines.chatterbox_official_23lang.streaming_model_manager import StreamingModelManager
        
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

