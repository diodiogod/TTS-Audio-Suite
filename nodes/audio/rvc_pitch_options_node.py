"""
RVC Pitch Extraction Options Node - Advanced pitch extraction settings for RVC Engine
Similar to F5-TTS Speech Editor Options, provides detailed control over pitch parameters
"""

import os
import sys
import importlib.util
from typing import Dict, Any

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


class RVCPitchOptionsNode(BaseTTSNode):
    """
    RVC Pitch Extraction Options Node - Advanced pitch extraction configuration.
    Provides detailed control over pitch extraction algorithms and parameters for RVC voice conversion.
    Similar to F5-TTS Speech Editor Options pattern.
    """
    
    @classmethod
    def NAME(cls):
        return "🔧 RVC Pitch Extraction Options"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available pitch extraction methods
        pitch_methods = [
            'rmvpe',      # Recommended - balanced quality/speed
            'rmvpe+',     # Enhanced RMVPE
            'mangio-crepe',   # Optimized Crepe
            'crepe',      # High quality but slower
            'pm',         # Praat-based, fast but basic
            'harvest',    # Traditional pitch extraction
            'dio',        # DIO algorithm
            'fcpe'        # Fast pitch extraction
        ]
        
        # Sample rates for resampling
        sample_rates = [0, 16000, 32000, 40000, 44100, 48000]
        
        return {
            "required": {
                # Core Pitch Extraction
                "pitch_detection": (pitch_methods, {
                    "default": "rmvpe",
                    "tooltip": "Pitch extraction algorithm:\n• RMVPE: Best balance of quality & speed (recommended)\n• RMVPE+: Enhanced RMVPE with better accuracy\n• Mangio-Crepe: Optimized Crepe, faster than standard\n• Crepe: Highest quality but slower processing\n• PM: Fast Praat-based extraction, basic quality\n• Harvest: Traditional method, good for speech\n• DIO: Fast algorithm, lower quality\n• FCPE: Very fast extraction for real-time use"
                }),
            },
            "optional": {
                # Advanced Parameters
                "crepe_hop_length": ("INT", {
                    "default": 160,
                    "min": 16,
                    "max": 512,
                    "step": 16,
                    "display": "slider",
                    "tooltip": "Only used by Crepe and Mangio-Crepe methods. Lower values track pitch changes more closely but cost more time and memory."
                }),
                
                # Processing Options
                "filter_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 7,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Only meaningful for Harvest smoothing in this backend. 0 means no extra smoothing; higher values smooth more but can flatten detail."
                }),
                "f0_autotune": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Snap the extracted pitch contour toward the nearest musical note after pitch detection. Can help singing or stylized output, but often sounds artificial on normal speech."
                }),
                
                # Performance Settings
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse cached loaded RVC/HuBERT models where possible for repeated processing. Usually worth leaving on."
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Crepe-family pitch batch size hint. Higher can be faster but uses more memory. Ignored by non-Crepe methods."
                })
            }
        }
    
    RETURN_TYPES = ("RVC_PITCH_OPTIONS",)
    RETURN_NAMES = ("rvc_pitch_options",)
    
    CATEGORY = "TTS Audio Suite/🎵 Audio Processing"
    
    FUNCTION = "create_pitch_options"
    
    DESCRIPTION = """
    RVC Pitch Extraction Options - Advanced pitch control for RVC voice conversion
    
    Provides detailed configuration of pitch extraction algorithms and advanced processing options.
    Connect to RVC Engine node for enhanced voice conversion control.
    
    Voice conversion quality parameters (index_rate, protect, rms_mix_rate) are configured in the RVC Engine node.
    
    Key Features:
    • Multiple pitch extraction algorithms (RMVPE, Crepe, PM, Harvest, etc.)
    • Advanced processing options (filtering, autotune, cache, batching)
    • Performance optimization (caching, batch processing, resampling)
    • Method-specific parameters (Crepe hop length, etc.)
    
    Pitch Methods Guide:
    • RMVPE: Best overall balance of quality and speed (recommended)
    • Crepe/Mangio-Crepe: Highest quality, slower processing
    • PM: Fastest, good for real-time applications
    • Harvest: Traditional algorithm, good for speech
    • DIO/FCPE: Alternative methods for specific use cases
    """
    
    def create_pitch_options(
        self,
        pitch_detection="rmvpe",
        crepe_hop_length=160,
        filter_radius=3,
        f0_autotune=False,
        use_cache=True,
        batch_size=1,
        **kwargs
    ):
        """
        Create RVC pitch extraction options configuration.
        
        Returns:
            Dictionary with pitch extraction parameters for RVC engine
        """
        try:
            # Validate parameters
            validated_options = {
                # Core pitch settings
                'f0_method': str(pitch_detection),
                'f0_autotune': bool(f0_autotune),
                
                # Advanced parameters
                'crepe_hop_length': max(16, min(512, int(crepe_hop_length))),
                'filter_radius': max(0, min(7, int(filter_radius))),
                
                # Performance settings
                'use_cache': bool(use_cache),
                'batch_size': max(1, min(8, int(batch_size)))
            }
            
            # Add method-specific parameters
            if 'crepe' in pitch_detection.lower():
                validated_options['crepe_hop_length'] = crepe_hop_length
            
            print(f"🔧 RVC Pitch Options: {pitch_detection} method")
            if f0_autotune:
                print("🎵 Autotune enabled")
                
            return (validated_options,)
            
        except Exception as e:
            print(f"❌ Error creating RVC pitch options: {e}")
            # Return default options on error
            default_options = {
                'f0_method': 'rmvpe',
                'f0_autotune': False,
                'crepe_hop_length': 160,
                'filter_radius': 3,
                'use_cache': True,
                'batch_size': 1,
                'error': str(e)
            }
            return (default_options,)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for RVC pitch options."""
        return True
