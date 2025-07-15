"""
F5-TTS Edit Node - Speech editing functionality
Enhanced Speech editing node using F5-TTS for targeted word/phrase replacement
"""

import torch
import numpy as np
import os
import tempfile
import torchaudio
from typing import Dict, Any, Optional, List, Tuple

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

from core.audio_processing import AudioProcessingUtils
from core.f5tts_edit_engine import F5TTSEditEngine
import comfy.model_management as model_management


class F5TTSEditNode(BaseF5TTSNode):
    """
    F5-TTS Speech editing node for targeted word/phrase replacement.
    Allows editing specific words/phrases in existing speech while maintaining voice characteristics.
    """
    
    @classmethod
    def NAME(cls):
        return "👄 F5-TTS Speech Editor"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_audio": ("AUDIO", {
                    "tooltip": "Original audio to edit"
                }),
                "original_text": ("STRING", {
                    "multiline": True,
                    "default": "Some call me nature, others call me mother nature.",
                    "tooltip": "Original text that matches the original audio"
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": "Some call me optimist, others call me realist.",
                    "tooltip": "Target text with desired changes"
                }),
                "edit_regions": ("STRING", {
                    "multiline": True,
                    "default": "1.42,2.44\n4.04,4.9",
                    "tooltip": "Edit regions as 'start,end' in seconds (one per line). These are the time regions to replace."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run F5-TTS model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "model": (["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"], {
                    "default": "F5TTS_v1_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
            },
            "optional": {
                "fix_durations": ("STRING", {
                    "multiline": True,
                    "default": "1.2\n1.0",
                    "tooltip": "Fixed durations for each edit region in seconds (one per line). Leave empty to use original durations."
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
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 71,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance. Values above 71 may cause ODE solver issues."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance strength. Controls how strictly F5-TTS follows the reference text. Higher values = more adherence to reference, lower values = more creative freedom."
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Sway sampling coefficient for F5-TTS inference. Controls the sampling behavior during generation. Negative values typically work better."
                }),
                "ode_method": (["euler", "midpoint"], {
                    "default": "euler",
                    "tooltip": "ODE solver method for F5-TTS inference. 'euler' is faster and typically sufficient, 'midpoint' may provide higher quality but slower generation."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("edited_audio", "edit_info")
    FUNCTION = "edit_speech"
    CATEGORY = "F5-TTS Voice"

    def __init__(self):
        super().__init__()
        self.current_model_name = "F5TTS_v1_Base"  # Default model name
        self.edit_engine = None
    
    def _parse_edit_regions(self, edit_regions_str: str) -> List[Tuple[float, float]]:
        """Parse edit regions from string format"""
        regions = []
        lines = edit_regions_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                try:
                    start, end = map(float, line.split(','))
                    regions.append((start, end))
                except ValueError:
                    raise ValueError(f"Invalid edit region format: '{line}'. Expected 'start,end' format.")
        return regions
    
    def _parse_fix_durations(self, fix_durations_str: str) -> Optional[List[float]]:
        """Parse fix durations from string format"""
        if not fix_durations_str.strip():
            return None
        
        durations = []
        lines = fix_durations_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                try:
                    duration = float(line)
                    durations.append(duration)
                except ValueError:
                    raise ValueError(f"Invalid fix duration format: '{line}'. Expected a number.")
        return durations
    
    def _get_edit_engine(self, device: str) -> F5TTSEditEngine:
        """Get or create the F5-TTS edit engine"""
        if self.edit_engine is None:
            self.edit_engine = F5TTSEditEngine(device, self.f5tts_sample_rate)
        return self.edit_engine
    
    def edit_speech(self, original_audio, original_text, target_text, edit_regions, 
                   device, model, seed, fix_durations="", temperature=0.8, speed=1.0, 
                   target_rms=0.1, nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, 
                   ode_method="euler"):
        
        def _process():
            # Validate inputs
            inputs = self.validate_inputs(
                original_audio=original_audio, original_text=original_text, target_text=target_text,
                edit_regions=edit_regions, device=device, model=model, seed=seed,
                fix_durations=fix_durations, temperature=temperature, speed=speed,
                target_rms=target_rms, nfe_step=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, ode_method=ode_method
            )
            
            # Load F5-TTS model
            self.load_f5tts_model(inputs["model"], inputs["device"])
            
            # Set seed for reproducibility
            self.set_seed(inputs["seed"])
            
            # Store model info for use in speech editing
            self.current_model_name = inputs["model"]
            
            # Parse edit regions and fix durations
            edit_regions_parsed = self._parse_edit_regions(inputs["edit_regions"])
            fix_durations_parsed = self._parse_fix_durations(inputs["fix_durations"])
            
            if fix_durations_parsed and len(fix_durations_parsed) != len(edit_regions_parsed):
                raise ValueError(f"Number of fix durations ({len(fix_durations_parsed)}) must match number of edit regions ({len(edit_regions_parsed)})")
            
            # Extract audio data
            if isinstance(original_audio, dict) and 'waveform' in original_audio:
                audio_tensor = original_audio['waveform']
                sample_rate = original_audio.get('sample_rate', self.f5tts_sample_rate)
            else:
                raise ValueError("Invalid audio format. Expected dictionary with 'waveform' key.")
            
            # Get edit engine and perform F5-TTS editing with compositing
            edit_engine = self._get_edit_engine(inputs["device"])
            edited_audio = edit_engine.perform_f5tts_edit(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                original_text=inputs["original_text"],
                target_text=inputs["target_text"],
                edit_regions=edit_regions_parsed,
                fix_durations=fix_durations_parsed,
                temperature=inputs["temperature"],
                speed=inputs["speed"],
                target_rms=inputs["target_rms"],
                nfe_step=inputs["nfe_step"],
                cfg_strength=inputs["cfg_strength"],
                sway_sampling_coef=inputs["sway_sampling_coef"],
                ode_method=inputs["ode_method"]
            )
            
            # Create edit info
            edit_info = f"F5-TTS Edit completed:\n"
            edit_info += f"Model: {inputs['model']}\n"
            edit_info += f"Original text: {inputs['original_text']}\n"
            edit_info += f"Target text: {inputs['target_text']}\n"
            edit_info += f"Edit regions: {edit_regions_parsed}\n"
            edit_info += f"Durations: {fix_durations_parsed}\n"
            edit_info += f"Output shape: {edited_audio.shape}\n"
            edit_info += f"Sample rate: {self.f5tts_sample_rate}Hz\n"
            edit_info += f"Device: {inputs['device']}\n"
            edit_info += f"Audio compositing: Enabled (preserves original quality outside edit regions)"
            
            # Return edited audio in ComfyUI format
            return (
                {"waveform": edited_audio, "sample_rate": self.f5tts_sample_rate},
                edit_info
            )
        
        return self.process_with_error_handling(_process)
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate inputs specific to speech editing"""
        # Call base validation
        validated = super(BaseF5TTSNode, self).validate_inputs(**inputs)
        
        # Additional validation for speech editing
        if not inputs.get("original_text", "").strip():
            raise ValueError("Original text cannot be empty")
        
        if not inputs.get("target_text", "").strip():
            raise ValueError("Target text cannot be empty")
        
        if not inputs.get("edit_regions", "").strip():
            raise ValueError("Edit regions cannot be empty")
        
        # Validate audio format
        original_audio = inputs.get("original_audio")
        if not isinstance(original_audio, dict) or 'waveform' not in original_audio:
            raise ValueError("Invalid audio format. Expected dictionary with 'waveform' key.")
        
        # Parse and validate edit regions format
        try:
            self._parse_edit_regions(inputs["edit_regions"])
        except ValueError as e:
            raise ValueError(f"Invalid edit regions format: {e}")
        
        # Parse and validate fix durations format if provided
        if inputs.get("fix_durations", "").strip():
            try:
                self._parse_fix_durations(inputs["fix_durations"])
            except ValueError as e:
                raise ValueError(f"Invalid fix durations format: {e}")
        
        return validated


# Node class mapping for ComfyUI registration
ChatterBoxF5TTSEditVoice = F5TTSEditNode

__all__ = ["ChatterBoxF5TTSEditVoice"]