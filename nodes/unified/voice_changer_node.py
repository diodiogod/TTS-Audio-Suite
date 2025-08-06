"""
Unified Voice Changer Node - Engine-agnostic voice conversion for TTS Audio Suite
Refactored from ChatterBox VC to support multiple engines (ChatterBox now, RVC in future)
"""

import torch
import tempfile
import os
import hashlib
from typing import Dict, Any

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
BaseVCNode = base_module.BaseVCNode

from utils.audio.processing import AudioProcessingUtils
import comfy.model_management as model_management


class UnifiedVoiceChangerNode(BaseVCNode):
    """
    Unified Voice Changer Node - Engine-agnostic voice conversion.
    Currently supports ChatterBox, prepared for future RVC and other voice conversion engines.
    Replaces ChatterBox VC node with engine-agnostic architecture.
    """
    
    @classmethod
    def NAME(cls):
        return "🔄 Voice Changer"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "TTS/VC engine configuration. For now, only ChatterBox supports voice conversion. Future engines like RVC will be supported."
                }),
                "source_audio": ("AUDIO", {
                    "tooltip": "The original voice audio you want to convert to sound like the target voice"
                }),
                "narrator_target": ("AUDIO", {
                    "tooltip": "The reference voice audio whose characteristics will be applied to the source audio (renamed from target_audio for consistency)"
                }),
                "refinement_passes": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Number of conversion iterations. Each pass refines the output to sound more like the target. Recommended: Max 5 passes - more can cause distortions. Each iteration is deterministic to reduce degradation."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("converted_audio", "conversion_info")
    FUNCTION = "convert_voice"
    CATEGORY = "TTS Audio Suite/Unified"

    def _create_proper_engine_node_instance(self, engine_data: Dict[str, Any]):
        """
        Create a proper engine VC node instance that has all the needed functionality.
        This preserves all existing VC functionality by creating instances of the original VC nodes.
        
        Args:
            engine_data: Engine configuration from TTS_engine input
            
        Returns:
            Proper engine VC node instance with all functionality
        """
        try:
            engine_type = engine_data.get("engine_type")
            config = engine_data.get("config", {})
            
            if engine_type == "chatterbox":
                # Import and create the original ChatterBox VC node
                from nodes.chatterbox.chatterbox_vc_node import ChatterboxVCNode
                engine_instance = ChatterboxVCNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                return engine_instance
                
            elif engine_type == "f5tts":
                # F5-TTS doesn't have voice conversion capability
                raise ValueError("F5-TTS engine does not support voice conversion. Use ChatterBox engine for voice conversion.")
                
            else:
                raise ValueError(f"Engine type '{engine_type}' does not support voice conversion. Currently supported: ChatterBox")
                
        except Exception as e:
            print(f"❌ Failed to create engine VC node instance: {e}")
            return None

    def convert_voice(self, TTS_engine: Dict[str, Any], source_audio: Dict[str, Any], 
                     narrator_target: Dict[str, Any], refinement_passes: int):
        """
        Convert voice using the selected engine.
        This is a DELEGATION WRAPPER that preserves all original VC functionality.
        
        Args:
            TTS_engine: Engine configuration from engine nodes
            source_audio: Source audio to convert
            narrator_target: Target voice characteristics (renamed for consistency)
            refinement_passes: Number of conversion iterations
            
        Returns:
            Tuple of (converted_audio, conversion_info)
        """
        try:
            # Validate engine input
            if not TTS_engine or not isinstance(TTS_engine, dict):
                raise ValueError("Invalid TTS_engine input - connect a TTS engine node")
            
            engine_type = TTS_engine.get("engine_type")
            config = TTS_engine.get("config", {})
            
            if not engine_type:
                raise ValueError("TTS engine missing engine_type")
            
            print(f"🔄 Voice Changer: Starting {engine_type} voice conversion")
            
            # Validate engine supports voice conversion
            if engine_type not in ["chatterbox"]:
                raise ValueError(f"Engine '{engine_type}' does not support voice conversion. Currently supported engines: ChatterBox")
            
            # Create proper engine VC node instance to preserve ALL functionality
            engine_instance = self._create_proper_engine_node_instance(TTS_engine)
            if not engine_instance:
                raise RuntimeError("Failed to create engine VC node instance")
            
            # Prepare parameters for the original VC node's convert_voice method
            if engine_type == "chatterbox":
                # ChatterBox VC parameters
                result = engine_instance.convert_voice(
                    source_audio=source_audio,
                    target_audio=narrator_target,  # Map narrator_target to target_audio for original node
                    refinement_passes=refinement_passes,
                    device=config.get("device", "auto")
                )
                
                # ChatterBox VC node returns only (converted_audio,)
                converted_audio = result[0]
                
                conversion_info = (
                    f"🔄 Voice Changer (Unified) - CHATTERBOX Engine:\n"
                    f"Refinement passes: {refinement_passes}\n"
                    f"Device: {config.get('device', 'auto')}\n"
                    f"Conversion completed successfully"
                )
                
            else:
                # Future engines (RVC, etc.) will be handled here
                raise ValueError(f"Engine type '{engine_type}' voice conversion not yet implemented")
            
            print(f"✅ Voice Changer: {engine_type} conversion successful")
            return (converted_audio, conversion_info)
                
        except Exception as e:
            error_msg = f"❌ Voice conversion failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Return empty audio and error info
            empty_audio = AudioProcessingUtils.create_silence_tensor(1.0, 24000)
            empty_comfy = AudioProcessingUtils.tensor_to_comfy_audio(empty_audio, 24000)
            
            return (empty_comfy, error_msg)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "UnifiedVoiceChangerNode": UnifiedVoiceChangerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedVoiceChangerNode": "🔄 Voice Changer"
}