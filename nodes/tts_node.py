"""
ChatterBox TTS Node - Migrated to use new foundation
Enhanced Text-to-Speech node using ChatterboxTTS with improved chunking
"""

import torch
import numpy as np
import os
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
import comfy.model_management as model_management


class ChatterboxTTSNode(BaseTTSNode):
    """
    Enhanced Text-to-Speech node using ChatterboxTTS - Voice Edition
    SUPPORTS BUNDLED CHATTERBOX + Enhanced Chunking
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
                    "default": "Hello! This is the enhanced ChatterboxTTS with bundled support and improved chunking. It can handle very long texts by intelligently splitting them into smaller segments."
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
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1, "control_after_generate": "fixed"}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "audio_prompt_path": ("STRING", {"default": ""}),
                # ENHANCED CHUNKING CONTROLS - ALL OPTIONAL FOR BACKWARD COMPATIBILITY
                "enable_chunking": ("BOOLEAN", {"default": True}),
                "max_chars_per_chunk": ("INT", {"default": 400, "min": 100, "max": 1000, "step": 50}),
                "chunk_combination_method": (["auto", "concatenate", "silence_padding", "crossfade"], {"default": "auto"}),
                "silence_between_chunks_ms": ("INT", {"default": 100, "min": 0, "max": 500, "step": 25}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        super().__init__()
        self.chunker = ImprovedChatterBoxChunker()

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
                       chunk_combination_method="auto", silence_between_chunks_ms=100):
        
        def _process():
            # Validate inputs
            inputs = self.validate_inputs(
                text=text, device=device, exaggeration=exaggeration,
                temperature=temperature, cfg_weight=cfg_weight, seed=seed,
                reference_audio=reference_audio, audio_prompt_path=audio_prompt_path,
                enable_chunking=enable_chunking, max_chars_per_chunk=max_chars_per_chunk,
                chunk_combination_method=chunk_combination_method,
                silence_between_chunks_ms=silence_between_chunks_ms
            )
            
            # Load model
            self.load_tts_model(inputs["device"])
            
            # Set seed for reproducibility
            self.set_seed(inputs["seed"])
            
            # Handle reference audio
            audio_prompt = self.handle_reference_audio(
                inputs.get("reference_audio"), inputs.get("audio_prompt_path", "")
            )
            
            # Determine if chunking is needed
            text_length = len(inputs["text"])
            
            if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                # Process single chunk
                wav = self.generate_tts_audio(
                    inputs["text"], audio_prompt, inputs["exaggeration"], 
                    inputs["temperature"], inputs["cfg_weight"]
                )
                model_source = self.model_manager.get_model_source("tts")
                info = f"Generated {wav.size(-1) / self.tts_model.sr:.1f}s audio from {text_length} characters (single chunk, {model_source} models)"
            else:
                # Split into chunks using improved chunker
                chunks = self.chunker.split_into_chunks(inputs["text"], inputs["max_chars_per_chunk"])
                
                # Process each chunk
                audio_segments = []
                for i, chunk in enumerate(chunks):
                    # Check for interruption
                    self.check_interruption(f"TTS generation chunk {i+1}/{len(chunks)}")
                    
                    # Show progress for multi-chunk generation
                    print(f"🎤 Generating TTS chunk {i+1}/{len(chunks)}...")
                    
                    chunk_audio = self.generate_tts_audio(
                        chunk, audio_prompt, inputs["exaggeration"], 
                        inputs["temperature"], inputs["cfg_weight"]
                    )
                    audio_segments.append(chunk_audio)
                
                # Combine audio segments
                wav = self.combine_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], text_length
                )
                
                # Generate info
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