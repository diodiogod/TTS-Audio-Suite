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

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


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
                "source_audio": (any_typ, {
                    "tooltip": "The original voice audio you want to convert to sound like the target voice. Accepts AUDIO input or Character Voices node output."
                }),
                "narrator_target": (any_typ, {
                    "tooltip": "The reference voice audio whose characteristics will be applied to the source audio. Accepts AUDIO input or Character Voices node output."
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
    CATEGORY = "TTS Audio Suite"

    def __init__(self):
        super().__init__()
        # Cache engine instances to prevent model reloading
        self._cached_engine_instances = {}

    def _extract_audio_from_input(self, audio_input, input_name: str):
        """
        Extract audio tensor from either AUDIO input or NARRATOR_VOICE input.
        
        Args:
            audio_input: Either AUDIO dict or NARRATOR_VOICE dict
            input_name: Name of input for error messages
            
        Returns:
            Audio dict suitable for voice conversion engines
        """
        try:
            if audio_input is None:
                raise ValueError(f"{input_name} input is required")
            
            # Check if it's a Character Voices node output (NARRATOR_VOICE)
            if isinstance(audio_input, dict) and "audio" in audio_input:
                # NARRATOR_VOICE input - extract the audio component
                audio_data = audio_input.get("audio")
                character_name = audio_input.get("character_name", "unknown")
                print(f"🔄 Voice Changer: Using {input_name} from Character Voices node ({character_name})")
                return audio_data
            
            # Check if it's a direct audio input (AUDIO)
            elif isinstance(audio_input, dict) and "waveform" in audio_input:
                # Direct AUDIO input
                print(f"🔄 Voice Changer: Using direct audio input for {input_name}")
                return audio_input
            
            else:
                raise ValueError(f"Invalid {input_name} format - expected AUDIO or Character Voices node output")
                
        except Exception as e:
            raise ValueError(f"Failed to process {input_name}: {e}")

    def _create_proper_engine_node_instance(self, engine_data: Dict[str, Any]):
        """
        Create a proper engine VC node instance that has all the needed functionality.
        Uses caching to reuse instances and preserve model state across conversions.
        
        Args:
            engine_data: Engine configuration from TTS_engine input
            
        Returns:
            Proper engine VC node instance with all functionality
        """
        try:
            engine_type = engine_data.get("engine_type")
            config = engine_data.get("config", {})
            
            # Create cache key based on engine type and stable config
            cache_key = f"{engine_type}_{hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8]}"
            
            # Check if we have a cached instance with the same configuration
            if cache_key in self._cached_engine_instances:
                cached_instance = self._cached_engine_instances[cache_key]
                print(f"🔄 Reusing cached {engine_type} VC engine instance (preserves model state)")
                return cached_instance
            
            if engine_type == "chatterbox":
                print(f"🔧 Creating new {engine_type} VC engine instance")
                
                # Import and create the original ChatterBox VC node using absolute import
                chatterbox_vc_path = os.path.join(nodes_dir, "chatterbox", "chatterbox_vc_node.py")
                chatterbox_vc_spec = importlib.util.spec_from_file_location("chatterbox_vc_module", chatterbox_vc_path)
                chatterbox_vc_module = importlib.util.module_from_spec(chatterbox_vc_spec)
                chatterbox_vc_spec.loader.exec_module(chatterbox_vc_module)
                
                ChatterboxVCNode = chatterbox_vc_module.ChatterboxVCNode
                engine_instance = ChatterboxVCNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance
                self._cached_engine_instances[cache_key] = engine_instance
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
            
            # Extract audio data from flexible inputs (support both AUDIO and NARRATOR_VOICE types)
            processed_source_audio = self._extract_audio_from_input(source_audio, "source_audio")
            processed_narrator_target = self._extract_audio_from_input(narrator_target, "narrator_target")
            
            # Create proper engine VC node instance to preserve ALL functionality
            engine_instance = self._create_proper_engine_node_instance(TTS_engine)
            if not engine_instance:
                raise RuntimeError("Failed to create engine VC node instance")
            
            # Prepare parameters for the original VC node's convert_voice method
            if engine_type == "chatterbox":
                # Extract language from engine config for multilingual VC support
                language = config.get("language", "English")
                print(f"🔄 Voice Changer: Using {language} language model for conversion")
                
                # ChatterBox VC parameters with language support
                result = engine_instance.convert_voice(
                    source_audio=processed_source_audio,
                    target_audio=processed_narrator_target,  # Map narrator_target to target_audio for original node
                    refinement_passes=refinement_passes,
                    device=config.get("device", "auto"),
                    language=language  # Pass language parameter to VC node
                )
                
                # ChatterBox VC node returns only (converted_audio,)
                converted_audio = result[0]
                
                # Get detailed model information for debugging
                model_source = "unknown"
                model_repo = "unknown"
                if hasattr(engine_instance, 'model_manager') and hasattr(engine_instance.model_manager, 'get_model_source'):
                    model_source = engine_instance.model_manager.get_model_source("vc") or "local/bundled"
                
                # Get repository information for HuggingFace models
                if model_source == "huggingface":
                    try:
                        from engines.chatterbox.language_models import get_model_config
                        model_config = get_model_config(language)
                        if model_config:
                            model_repo = model_config.get("repo", "unknown")
                        else:
                            model_repo = "ResembleAI/chatterbox"  # Default English repo
                    except ImportError:
                        model_repo = "ResembleAI/chatterbox"  # Fallback
                
                conversion_info = (
                    f"🔄 Voice Changer (Unified) - CHATTERBOX Engine:\n"
                    f"Language Model: {language}\n"
                    f"Model Source: {model_source}\n"
                    + (f"Repository: {model_repo}\n" if model_source == "huggingface" else "") +
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
            empty_audio = AudioProcessingUtils.create_silence(1.0, 24000)
            empty_comfy = AudioProcessingUtils.format_for_comfyui(empty_audio, 24000)
            
            return (empty_comfy, error_msg)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "UnifiedVoiceChangerNode": UnifiedVoiceChangerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedVoiceChangerNode": "🔄 Voice Changer"
}