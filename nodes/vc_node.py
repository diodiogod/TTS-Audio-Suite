"""
ChatterBox Voice Conversion Node - Migrated to use new foundation
Voice Conversion node using ChatterboxVC with improved architecture
"""

import torch
import tempfile
import os
from typing import Dict, Any

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
BaseVCNode = base_module.BaseVCNode

import torchaudio


class ChatterboxVCNode(BaseVCNode):
    """
    Voice Conversion node using ChatterboxVC - Voice Edition
    SUPPORTS BUNDLED CHATTERBOX
    """
    
    @classmethod
    def NAME(cls):
        return "🔄 ChatterBox Voice Conversion (diogod)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("converted_audio",)
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        super().__init__()

    def prepare_audio_files(self, source_audio: Dict[str, Any], target_audio: Dict[str, Any]) -> tuple[str, str]:
        """
        Prepare audio files for voice conversion by saving to temporary files.
        
        Args:
            source_audio: Source audio dictionary from ComfyUI
            target_audio: Target audio dictionary from ComfyUI
            
        Returns:
            Tuple of (source_path, target_path)
        """
        # Save source audio to temporary file
        source_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        source_temp.close()
        
        source_waveform = source_audio["waveform"]
        if source_waveform.dim() == 3:
            source_waveform = source_waveform.squeeze(0)  # Remove batch dimension
        
        torchaudio.save(source_temp.name, source_waveform.cpu(), source_audio["sample_rate"])
        self._temp_files.append(source_temp.name)
        
        # Save target audio to temporary file
        target_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        target_temp.close()
        
        target_waveform = target_audio["waveform"]
        if target_waveform.dim() == 3:
            target_waveform = target_waveform.squeeze(0)  # Remove batch dimension
        
        torchaudio.save(target_temp.name, target_waveform.cpu(), target_audio["sample_rate"])
        self._temp_files.append(target_temp.name)
        
        return source_temp.name, target_temp.name

    def convert_voice(self, source_audio, target_audio, device):
        """
        Perform voice conversion using the loaded model.
        
        Args:
            source_audio: Source audio from ComfyUI
            target_audio: Target voice audio from ComfyUI
            device: Target device
            
        Returns:
            Converted audio in ComfyUI format
        """
        def _process():
            # Load model
            self.load_vc_model(device)
            
            # Prepare audio files
            source_path, target_path = self.prepare_audio_files(source_audio, target_audio)
            
            try:
                # Perform voice conversion
                wav = self.vc_model.generate(
                    source_path,
                    target_voice_path=target_path
                )
                
                # Return audio in ComfyUI format
                return (self.format_audio_output(wav, self.vc_model.sr),)
                
            finally:
                # Cleanup is handled automatically by the base class destructor
                # and the cleanup_temp_files method
                pass
        
        return self.process_with_error_handling(_process)