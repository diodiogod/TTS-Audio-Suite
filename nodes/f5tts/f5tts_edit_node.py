"""
F5-TTS Edit Node - Speech editing functionality
Modularized version using exact working implementation
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

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(nodes_dir, "base", "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from utils.audio.processing import AudioProcessingUtils
from engines.f5tts.f5tts_edit_engine import F5TTSEditEngine
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
                "model": (BaseF5TTSNode.get_available_models_for_dropdown(), {
                    "default": "F5TTS_v1_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
            },
            "optional": {
                "edit_options": ("F5TTS_EDIT_OPTIONS", {
                    "tooltip": "Optional advanced editing options"
                }),
                "fix_durations": ("STRING", {
                    "multiline": True,
                    "default": "1.2\n1.0",
                    "tooltip": "Fixed durations for each edit region in seconds (one per line). Leave empty to use original durations."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness in F5-TTS generation. Higher values = more creative/varied speech, lower values = more consistent/predictable speech."
                }),
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 71,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance. Values above 71 may cause ODE solver issues."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Speech generation control. Lower values (1.0-1.5) = more natural, conversational delivery. Higher values (3.0-5.0) = crisper, more articulated speech with stronger emphasis. Default 2.0 balances naturalness and clarity."
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
    CATEGORY = "TTS Audio Suite/👄 F5-TTS"

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
                   device, model, seed, edit_options=None, fix_durations="", temperature=0.8, 
                   nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, 
                   ode_method="euler"):
        
        def _process():
            # Validate inputs
            inputs = self.validate_inputs(
                original_audio=original_audio, original_text=original_text, target_text=target_text,
                edit_regions=edit_regions, device=device, model=model, seed=seed,
                fix_durations=fix_durations, temperature=temperature,
                nfe_step=nfe_step, cfg_strength=cfg_strength,
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
            edit_engine = self._get_edit_engine(self.device)  # Use resolved device from base node
            edited_audio = edit_engine.perform_f5tts_edit(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                original_text=inputs["original_text"],
                target_text=inputs["target_text"],
                edit_regions=edit_regions_parsed,
                fix_durations=fix_durations_parsed,
                temperature=inputs["temperature"],
                nfe_step=inputs["nfe_step"],
                cfg_strength=inputs["cfg_strength"],
                sway_sampling_coef=inputs["sway_sampling_coef"],
                ode_method=inputs["ode_method"],
                seed=inputs["seed"],
                current_model_name=self.current_model_name,
                edit_options=edit_options,
                unified_model=self.f5tts_model  # Pass the unified model
            )
            
            # Generate detailed info
            total_duration = edited_audio.size(-1) / self.f5tts_sample_rate
            original_duration = audio_tensor.size(-1) / sample_rate
            model_info = self.get_f5tts_model_info()
            
            # Build detailed edit region info
            region_info = []
            for i, (start, end) in enumerate(edit_regions_parsed):
                if fix_durations_parsed and i < len(fix_durations_parsed):
                    fixed_dur = fix_durations_parsed[i]
                    region_info.append(f"Region {i+1}: {start:.2f}-{end:.2f}s (orig {end-start:.2f}s) -> fixed {fixed_dur:.2f}s")
                else:
                    region_info.append(f"Region {i+1}: {start:.2f}-{end:.2f}s ({end-start:.2f}s)")
            
            edit_info = (f"F5-TTS Edit Complete:\n"
                        f"Duration: {original_duration:.1f}s -> {total_duration:.1f}s\n"
                        f"Model: {model_info.get('model_name', 'unknown')}\n"
                        f"Edit Regions ({len(edit_regions_parsed)}):\n" + "\n".join(region_info) + "\n"
                        f"Original: '{inputs['original_text'][:80]}{'...' if len(inputs['original_text']) > 80 else ''}'\n"
                        f"Target: '{inputs['target_text'][:80]}{'...' if len(inputs['target_text']) > 80 else ''}'\n"
                        f"Audio Compositing: Enabled (preserves original quality outside edit regions)")
            
            # Return audio in ComfyUI format
            return (
                self.format_f5tts_audio_output(edited_audio),
                edit_info
            )
        
        return self.process_with_error_handling(_process)
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate inputs specific to speech editing"""
        # Call base validation
        validated = super(BaseF5TTSNode, self).validate_inputs(**inputs)
        
        # Validate required inputs
        if not validated.get("original_text", "").strip():
            raise ValueError("Original text is required and cannot be empty")
        
        if not validated.get("target_text", "").strip():
            raise ValueError("Target text is required and cannot be empty")
        
        if not validated.get("edit_regions", "").strip():
            raise ValueError("Edit regions are required and cannot be empty")
        
        # Validate edit regions format
        try:
            edit_regions = self._parse_edit_regions(validated["edit_regions"])
            if not edit_regions:
                raise ValueError("At least one edit region must be specified")
        except ValueError as e:
            raise ValueError(f"Invalid edit regions: {e}")
        
        # Validate fix durations if provided
        fix_durations_str = validated.get("fix_durations", "").strip()
        if fix_durations_str:
            try:
                fix_durations = self._parse_fix_durations(fix_durations_str)
                if fix_durations and len(fix_durations) != len(edit_regions):
                    raise ValueError(f"Number of fix durations ({len(fix_durations)}) must match number of edit regions ({len(edit_regions)})")
            except ValueError as e:
                raise ValueError(f"Invalid fix durations: {e}")
        
        return validated


# Node class mapping for ComfyUI registration
ChatterBoxF5TTSEditVoice = F5TTSEditNode

__all__ = ["ChatterBoxF5TTSEditVoice"]