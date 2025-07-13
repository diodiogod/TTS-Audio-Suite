"""
F5-TTS Node - Basic text-to-speech generation
Enhanced Text-to-Speech node using F5-TTS with reference audio + text
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

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(current_dir, "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from core.text_chunking import ImprovedChatterBoxChunker
from core.audio_processing import AudioProcessingUtils
import comfy.model_management as model_management


class F5TTSNode(BaseF5TTSNode):
    """
    Basic F5-TTS text-to-speech generation node.
    Requires reference audio + text for voice cloning.
    """
    
    @classmethod
    def NAME(cls):
        return "🎤 F5-TTS Voice Generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import os
        from pathlib import Path
        
        # Get available reference audio files with companion .txt files from models/voices/
        models_dir = folder_paths.models_dir
        voices_dir = os.path.join(models_dir, "voices")
        reference_files = ["none"]  # Add "none" option for manual input
        
        if os.path.exists(voices_dir):
            voice_files = folder_paths.filter_files_content_types(
                os.listdir(voices_dir), ["audio", "video"]
            )
            for file in voice_files:
                # Check if companion .txt file exists
                full_file_path = os.path.join(voices_dir, file)
                txt_file = cls._get_companion_txt_file(full_file_path)
                if os.path.isfile(txt_file):
                    reference_files.append(file)
        
        reference_files = sorted(reference_files)
        
        # Node layout with opt_reference_text as second widget
        base_types = {
            "required": {
                "reference_audio_file": (reference_files, {
                    "default": "none",
                    "tooltip": "Reference voice from models/voices/ folder (with companion .txt file). Select 'none' to use direct inputs below."
                }),
                "opt_reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Direct reference text input (required when using opt_reference_audio)."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run F5-TTS model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "model": (["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"], {
                    "default": "F5TTS_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is F5-TTS integrated with ChatterBox Voice. It provides high-quality text-to-speech with voice cloning capabilities using reference audio and text.",
                    "tooltip": "The text to convert to speech using F5-TTS."
                }),
            },
            "optional": {
                "opt_reference_audio": ("AUDIO", {
                    "tooltip": "Direct reference audio input (used when reference_audio_file is 'none')"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness in F5-TTS generation. Higher values = more creative/varied speech, lower values = more consistent/predictable speech."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "F5-TTS native speech speed control. 1.0 = normal speed, 0.5 = half speed (slower), 2.0 = double speed (faster)."
                }),
                "target_rms": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Target audio volume level (Root Mean Square). Controls output loudness normalization. Higher values = louder audio output."
                }),
                "cross_fade_duration": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Duration in seconds for smooth audio transitions between F5-TTS segments. Prevents audio clicks/pops by blending segment boundaries."
                }),
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 100,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance strength. Controls how strictly F5-TTS follows the reference text. Higher values = more adherence to reference, lower values = more creative freedom."
                }),
                "enable_chunking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable text chunking for long texts. When enabled, long texts are split into smaller chunks for more stable generation."
                }),
                "max_chars_per_chunk": ("INT", {
                    "default": 400, "min": 100, "max": 1000, "step": 50,
                    "tooltip": "Maximum characters per chunk when chunking is enabled. Smaller chunks = more stable but potentially less coherent speech."
                }),
                "chunk_combination_method": (["auto", "concatenate", "silence_padding", "crossfade"], {
                    "default": "auto",
                    "tooltip": "Method to combine audio chunks: 'auto' chooses best method, 'concatenate' joins directly, 'silence_padding' adds silence between chunks, 'crossfade' smoothly blends chunks."
                }),
                "silence_between_chunks_ms": ("INT", {
                    "default": 100, "min": 0, "max": 500, "step": 25,
                    "tooltip": "Silence duration between chunks in milliseconds when using 'silence_padding' combination method. Longer silences = more distinct separation between chunks."
                }),
            }
        }
        
        return base_types

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "F5-TTS Voice"

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def _get_companion_txt_file(audio_file_path):
        """Get the path to companion .txt file for an audio file"""
        from pathlib import Path
        p = Path(audio_file_path)
        return os.path.join(os.path.dirname(audio_file_path), p.stem + ".txt")
    
    def _load_reference_from_file(self, reference_audio_file):
        """Load reference audio and text from models/voices/ folder"""
        import folder_paths
        
        if reference_audio_file == "none":
            return None, None
        
        # Get full audio file path from models/voices/
        models_dir = folder_paths.models_dir
        voices_dir = os.path.join(models_dir, "voices")
        audio_path = os.path.join(voices_dir, reference_audio_file)
        
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        
        # Get companion text file
        txt_file = self._get_companion_txt_file(audio_path)
        
        if not os.path.isfile(txt_file):
            raise FileNotFoundError(f"Companion text file not found: {txt_file}")
        
        # Read reference text
        with open(txt_file, 'r', encoding='utf-8') as file:
            ref_text = file.read().strip()
        
        return audio_path, ref_text
    
    def _handle_reference_with_priority_chain(self, inputs):
        """Handle reference audio and text with improved priority chain"""
        reference_audio_file = inputs.get("reference_audio_file", "none")
        opt_reference_audio = inputs.get("opt_reference_audio")
        opt_reference_text = inputs.get("opt_reference_text", "").strip()
        
        # PRIORITY 1: Check reference_audio_file first
        if reference_audio_file != "none":
            try:
                audio_path, auto_ref_text = self._load_reference_from_file(reference_audio_file)
                if audio_path and auto_ref_text:
                    print(f"✅ F5-TTS: Using reference file '{reference_audio_file}' with auto-detected text")
                    return audio_path, auto_ref_text
            except Exception as e:
                print(f"⚠️ F5-TTS: Failed to load reference file '{reference_audio_file}': {e}")
                print("🔄 F5-TTS: Falling back to manual inputs...")
        
        # PRIORITY 2: Use opt_reference_audio + opt_reference_text (both required)
        if opt_reference_audio is not None:
            # Handle the audio input to get file path
            audio_prompt = self.handle_reference_audio(opt_reference_audio, "")
            
            if audio_prompt:
                # Check if opt_reference_text is provided
                if opt_reference_text and opt_reference_text.strip():
                    print(f"📝 F5-TTS: Using direct reference audio + text inputs")
                    return audio_prompt, opt_reference_text.strip()
                
                # Error - audio provided but no text
                raise ValueError(
                    "F5-TTS requires reference text. Please connect text to opt_reference_text input."
                )
        
        # FINAL: No reference inputs provided at all
        raise ValueError(
            "F5-TTS requires reference audio and text. Please provide either:\n"
            "1. Select a reference_audio_file with companion .txt file, OR\n"
            "2. Connect opt_reference_audio input and provide opt_reference_text"
        )
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate and normalize inputs."""
        # Call the base class validate_inputs directly (BaseChatterBoxNode)
        validated = super(BaseF5TTSNode, self).validate_inputs(**inputs)
        
        # Skip the F5-TTS base validation since we handle reference validation in our priority chain
        # Don't call validate_f5tts_inputs since it expects ref_text which we don't have
        
        # Handle None/empty values for backward compatibility
        if validated.get("enable_chunking") is None:
            validated["enable_chunking"] = True
        if validated.get("max_chars_per_chunk") is None or validated.get("max_chars_per_chunk", 0) < 100:
            validated["max_chars_per_chunk"] = 400
        if not validated.get("chunk_combination_method"):
            validated["chunk_combination_method"] = "auto"
        if validated.get("silence_between_chunks_ms") is None:
            validated["silence_between_chunks_ms"] = 100
        
        # Validate model name if provided
        model_name = validated.get("model", "F5TTS_Base")
        try:
            available_models = self._get_available_models()
            if model_name not in available_models:
                print(f"⚠️ F5-TTS: Model '{model_name}' not in available list, but will attempt to load")
        except:
            pass  # Don't fail validation on model check
        
        return validated
    
    def _get_available_models(self):
        """Get list of available F5-TTS models"""
        try:
            from chatterbox.f5tts.f5tts import get_f5tts_models
            return get_f5tts_models()
        except ImportError:
            return ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]
    
    def generate_speech(self, reference_audio_file, text, device, model, seed,
                       opt_reference_audio=None, opt_reference_text="",
                       audio_prompt_path="", enable_chunking=True, max_chars_per_chunk=400,
                       chunk_combination_method="auto", silence_between_chunks_ms=100,
                       temperature=0.8, speed=1.0, target_rms=0.1,
                       cross_fade_duration=0.15, nfe_step=32, cfg_strength=2.0):
        
        def _process():
            # Validate inputs
            inputs = self.validate_inputs(
                reference_audio_file=reference_audio_file, text=text, device=device, model=model, seed=seed,
                opt_reference_audio=opt_reference_audio, opt_reference_text=opt_reference_text,
                audio_prompt_path=audio_prompt_path, enable_chunking=enable_chunking,
                max_chars_per_chunk=max_chars_per_chunk, chunk_combination_method=chunk_combination_method,
                silence_between_chunks_ms=silence_between_chunks_ms, temperature=temperature,
                speed=speed, target_rms=target_rms, cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step, cfg_strength=cfg_strength
            )
            
            # Load F5-TTS model
            self.load_f5tts_model(inputs["model"], inputs["device"])
            
            # Set seed for reproducibility
            self.set_seed(inputs["seed"])
            
            # Handle reference audio and text with priority chain
            audio_prompt, validated_ref_text = self._handle_reference_with_priority_chain(inputs)
            
            # Determine if chunking is needed
            text_length = len(inputs["text"])
            
            if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                # Process single chunk
                wav = self.generate_f5tts_audio(
                    text=inputs["text"],
                    ref_audio_path=audio_prompt,
                    ref_text=validated_ref_text,
                    temperature=inputs["temperature"],
                    speed=inputs["speed"],
                    target_rms=inputs["target_rms"],
                    cross_fade_duration=inputs["cross_fade_duration"],
                    nfe_step=inputs["nfe_step"],
                    cfg_strength=inputs["cfg_strength"]
                )
                model_info = self.get_f5tts_model_info()
                info = f"Generated {wav.size(-1) / self.f5tts_sample_rate:.1f}s audio from {text_length} characters (single chunk, F5-TTS {model_info.get('model_name', 'unknown')})"
            else:
                # Split into chunks using improved chunker
                chunks = self.chunker.split_into_chunks(inputs["text"], inputs["max_chars_per_chunk"])
                
                # Process each chunk
                audio_segments = []
                for i, chunk in enumerate(chunks):
                    # Check for interruption
                    self.check_interruption(f"F5-TTS generation chunk {i+1}/{len(chunks)}")
                    
                    # Show progress for multi-chunk generation
                    print(f"🎤 Generating F5-TTS chunk {i+1}/{len(chunks)}...")
                    
                    chunk_audio = self.generate_f5tts_audio(
                        text=chunk,
                        ref_audio_path=audio_prompt,
                        ref_text=validated_ref_text,
                        temperature=inputs["temperature"],
                        speed=inputs["speed"],
                        target_rms=inputs["target_rms"],
                        cross_fade_duration=inputs["cross_fade_duration"],
                        nfe_step=inputs["nfe_step"],
                        cfg_strength=inputs["cfg_strength"]
                    )
                    audio_segments.append(chunk_audio)
                
                # Combine audio segments
                wav = self.combine_f5tts_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], text_length
                )
                
                # Generate info
                total_duration = wav.size(-1) / self.f5tts_sample_rate
                avg_chunk_size = text_length // len(chunks)
                model_info = self.get_f5tts_model_info()
                info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, F5-TTS {model_info.get('model_name', 'unknown')})"
            
            # Return audio in ComfyUI format
            return (
                self.format_f5tts_audio_output(wav),
                info
            )
        
        return self.process_with_error_handling(_process)