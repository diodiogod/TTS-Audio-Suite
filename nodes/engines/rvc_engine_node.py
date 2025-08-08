"""
RVC Engine Node - Unified RVC configuration for TTS Audio Suite
Consolidates functionality from multiple reference RVC nodes into single interface
Combines RVC Model, Hubert Model, and Voice Changer parameters
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Tuple

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

from engines.adapters.rvc_adapter import RVCEngineAdapter


class RVCEngineNode(BaseTTSNode):
    """
    RVC Engine configuration node.
    Consolidates RVC model loading, Hubert model loading, and core voice conversion parameters
    into single user-friendly interface following TTS Suite patterns.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ RVC Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        try:
            # Get available models through RVC adapter
            adapter = RVCEngineAdapter()
            available_models = adapter.get_available_models()
            pitch_methods = adapter.get_pitch_extraction_methods()
            
            rvc_models = available_models.get('rvc_models', ["No RVC models found"])
            hubert_models = available_models.get('hubert_models', ["content-vec-best.safetensors"])
            
            # Add sample rates for resampling
            sample_rates = [0, 16000, 32000, 40000, 44100, 48000]
            
        except ImportError as e:
            print(f"Warning: Could not load RVC adapter: {e}")
            rvc_models = ["No RVC models found"]
            hubert_models = ["content-vec-best.safetensors"]
            pitch_methods = ['rmvpe', 'crepe', 'mangio-crepe', 'rmvpe+']
            sample_rates = [0, 16000, 32000, 40000, 44100, 48000]
        
        return {
            "required": {
                # Model Selection (consolidates LoadRVCModel and LoadHubertModel)
                "rvc_model": (rvc_models, {
                    "default": rvc_models[0] if rvc_models else "No RVC models found",
                    "tooltip": "RVC voice model (.pth file) for voice conversion. Place models in ComfyUI/models/RVC/ directory."
                }),
                "hubert_model": (hubert_models, {
                    "default": hubert_models[0] if hubert_models else "content-vec-best.safetensors",
                    "tooltip": "Hubert feature extraction model. content-vec-best is recommended for most use cases."
                }),
                
                # Core Voice Conversion Parameters (from Voice Changer node)
                "pitch_shift": ("INT", {
                    "default": 0,
                    "min": -14,
                    "max": 14,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Pitch shift in semitones. 0=no change, +12=octave up (male→female), -12=octave down (female→male)"
                }),
                "f0_method": (pitch_methods, {
                    "default": "rmvpe",
                    "tooltip": "Pitch extraction algorithm. RMVPE=balanced quality/speed, Crepe=highest quality, Mangio-Crepe=optimized"
                }),
                "index_rate": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Index file influence (0.0-1.0). Higher=more like training voice, lower=more like input voice"
                }),
                "protect": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider", 
                    "tooltip": "Consonant protection (0.0-0.5). Protects speech clarity, higher=more protection"
                }),
                "rms_mix_rate": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Volume envelope mixing (0.0-1.0). Controls how much of original volume envelope to preserve"
                }),
            },
            "optional": {
                # Advanced Pitch Options
                "rvc_pitch_options": ("RVC_PITCH_OPTIONS", {
                    "tooltip": "Optional advanced pitch extraction settings from RVC Pitch Options node. Overrides basic parameters."
                }),
                
                # Optional advanced parameters
                "index_file": ("STRING", {
                    "default": "",
                    "tooltip": "Optional .index file path for enhanced voice matching. Leave empty for auto-detection."
                }),
                "f0_autotune": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable automatic pitch correction/tuning"
                }),
                "resample_sr": (sample_rates, {
                    "default": 0,
                    "tooltip": "Output sample rate (0=use input rate). 44100/48000 recommended for high quality"
                }),
                
                # Model Quality Settings
                "use_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache conversions for faster repeated processing"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Processing device. Auto=optimal device detection"
                })
            }
        }
    
    RETURN_TYPES = ("RVC_ENGINE",)
    RETURN_NAMES = ("rvc_engine",)
    
    CATEGORY = "🎵 TTS Audio Suite/Engines"
    
    FUNCTION = "create_engine"
    
    DESCRIPTION = """
    RVC Engine - Real-time Voice Conversion
    
    Consolidates RVC model loading and voice conversion parameters into unified interface.
    Supports pitch shifting, advanced quality controls, and multiple pitch extraction algorithms.
    
    Key Features:
    • Voice conversion with pitch control
    • Multiple pitch extraction algorithms (RMVPE, Crepe, etc.)
    • Quality controls (index rate, consonant protection)
    • Automatic model management and caching
    • Compatible with unified Voice Changer node
    """
    
    def create_engine(
        self,
        rvc_model,
        hubert_model, 
        pitch_shift=0,
        f0_method="rmvpe",
        index_rate=0.75,
        protect=0.25,
        rms_mix_rate=0.25,
        rvc_pitch_options=None,
        index_file="",
        f0_autotune=False,
        resample_sr=0,
        use_cache=True,
        device="auto"
    ):
        """
        Create RVC engine adapter with loaded models and parameters.
        
        Returns:
            RVC engine adapter configured for voice conversion
        """
        try:
            # Create RVC adapter
            adapter = RVCEngineAdapter()
            
            # Merge pitch options if provided (advanced options override basic parameters)
            final_pitch_params = {
                'f0_method': f0_method,
                'f0_autotune': f0_autotune,
                'index_rate': index_rate,
                'protect': protect,
                'rms_mix_rate': rms_mix_rate,
                'resample_sr': resample_sr,
                'use_cache': use_cache
            }
            
            if rvc_pitch_options:
                # Advanced pitch options override basic parameters
                if isinstance(rvc_pitch_options, dict):
                    final_pitch_params.update(rvc_pitch_options)
                    print("🎛️ Using advanced pitch options from RVC Pitch Options node")
                else:
                    print("⚠️  Invalid pitch options format, using basic parameters")
            
            # Resolve device
            if device == "auto":
                import comfy.model_management as model_management
                device = str(model_management.get_torch_device())
            
            # Load models through adapter
            try:
                import folder_paths
                models_dir = folder_paths.models_dir
                
                # Resolve model paths
                if rvc_model.startswith("RVC/"):
                    rvc_model_path = os.path.join(models_dir, rvc_model)
                else:
                    rvc_model_path = os.path.join(models_dir, "RVC", rvc_model)
                
                hubert_model_path = os.path.join(models_dir, hubert_model)
                
                # Handle optional index file
                index_path = None
                if index_file:
                    if os.path.isabs(index_file):
                        index_path = index_file
                    else:
                        # Look in RVC/.index directory
                        index_path = os.path.join(models_dir, "RVC", ".index", index_file)
                        if not os.path.exists(index_path):
                            print(f"Warning: Index file not found: {index_path}")
                            index_path = None
                
                # Load models
                model_info = adapter.load_models(
                    rvc_model_path=rvc_model_path,
                    hubert_model_path=hubert_model_path,
                    index_path=index_path
                )
                
                # Store configuration in adapter
                adapter.config = {
                    'model_key': model_info['model_key'],
                    'pitch_shift': pitch_shift,
                    'device': device,
                    'rvc_model': rvc_model,
                    'hubert_model': hubert_model,
                    **final_pitch_params  # Include all final pitch parameters
                }
                
                print(f"RVC Engine created - Model: {os.path.basename(rvc_model)}, Hubert: {os.path.basename(hubert_model)}")
                print(f"Pitch method: {final_pitch_params['f0_method']}, Pitch shift: {pitch_shift}, Device: {device}")
                if rvc_pitch_options:
                    print("🎛️ Advanced pitch options applied")
                
                return (adapter,)
                
            except ImportError:
                print("Warning: folder_paths not available, using relative paths")
                # Fallback without folder_paths
                model_info = adapter.load_models(
                    rvc_model_path=rvc_model,
                    hubert_model_path=hubert_model,
                    index_path=index_file if index_file else None
                )
                
                adapter.config = {
                    'model_key': model_info['model_key'],
                    'pitch_shift': pitch_shift,
                    'f0_method': f0_method,
                    'index_rate': index_rate,
                    'protect': protect,
                    'rms_mix_rate': rms_mix_rate,
                    'f0_autotune': f0_autotune,
                    'resample_sr': resample_sr,
                    'use_cache': use_cache,
                    'device': device,
                    'rvc_model': rvc_model,
                    'hubert_model': hubert_model
                }
                
                return (adapter,)
        
        except Exception as e:
            print(f"Error creating RVC engine: {e}")
            # Return a basic adapter even on error for graceful degradation
            adapter = RVCEngineAdapter()
            adapter.config = {
                'error': str(e),
                'rvc_model': rvc_model,
                'hubert_model': hubert_model
            }
            return (adapter,)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for RVC engine creation."""
        return True