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
    RVC Engine configuration node - VOICE CONVERSION ONLY.
    
    ⚠️ IMPORTANT: RVC is for voice conversion only - it converts existing audio to a different voice.
    This engine CANNOT generate speech from text and will NOT work with TTS Text or TTS SRT nodes.
    Use only with the Voice Changer node to convert audio files or microphone recordings.
    
    Consolidates RVC model loading, Hubert model loading, and core voice conversion parameters
    into single user-friendly interface following TTS Suite patterns.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ RVC Engine"
    
    @classmethod
    def _get_available_rvc_models(cls) -> list:
        """Get available RVC models without importing heavy modules.
        Reconstructs the discovery logic using file reading only."""
        rvc_models = []

        try:
            import folder_paths

            # Look for RVC models - try TTS path first, then legacy
            rvc_search_paths = [
                os.path.join(folder_paths.models_dir, "TTS", "RVC"),
                os.path.join(folder_paths.models_dir, "RVC")  # Legacy
            ]

            for rvc_dir in rvc_search_paths:
                if os.path.exists(rvc_dir):
                    try:
                        for file in os.listdir(rvc_dir):
                            if file.endswith('.pth'):
                                relative_path = os.path.relpath(os.path.join(rvc_dir, file), folder_paths.models_dir)
                                rvc_models.append(relative_path)
                    except OSError:
                        continue
        except Exception:
            pass

        return rvc_models if rvc_models else ["No RVC models found"]

    @classmethod
    def INPUT_TYPES(cls):
        rvc_models = cls._get_available_rvc_models()

        # Static HuBERT model definitions (not heavy)
        HUBERT_MODELS = {
            "auto": "Automatically select best available model",
            "content-vec-best": "Content Vec 768 (Recommended)",
            "hubert-base-japanese": "HuBERT Japanese",
            "hubert-base-korean": "HuBERT Korean",
            "chinese-hubert-base": "Chinese HuBERT Base",
            "hubert-large": "HuBERT Large (Highest Quality)"
        }
        hubert_models = [f"{key}: {desc}" for key, desc in HUBERT_MODELS.items()]

        # Static pitch methods (not heavy)
        pitch_methods = ['rmvpe', 'crepe', 'mangio-crepe', 'rmvpe+']

        # Add sample rates for resampling
        sample_rates = [0, 16000, 32000, 40000, 44100, 48000]
        
        return {
            "required": {
                # Core Voice Conversion Parameters
                "pitch": ("INT", {
                    "default": 0,
                    "min": -14,
                    "max": 14,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Pitch shift in semitones. 0=no change, +12=octave up (male→female), -12=octave down (female→male)"
                }),
                "index_ratio": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Index file influence (0.0-1.0). Higher=more like training voice, lower=more like input voice"
                }),
                "consonant_protection": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider", 
                    "tooltip": "Consonant protection - Protects speech clarity. Low=voice changes more, High=keeps original pronunciation clearer"
                }),
                "volume_envelope": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Volume envelope mixing - Controls volume patterns. Low=use target voice volume, High=keep original voice volume patterns"
                }),
            },
            "optional": {
                # HuBERT Model Selection
                "hubert_model": (hubert_models, {
                    "default": hubert_models[0] if hubert_models else "auto: Automatically select best available model",
                    "tooltip": """HuBERT Model for feature extraction:

• Auto: Automatically select the best available model based on your language
• Content Vec 768: RECOMMENDED - Best for RVC voice conversion, fastest loading
• HuBERT Japanese: Optimized for Japanese voices and phonetics  
• HuBERT Korean: Specialized for Korean speech patterns
• Chinese HuBERT: Fine-tuned for Mandarin Chinese tonal patterns
• HuBERT Large: Highest quality but slower processing

Models will auto-download if not present. Choose language-specific models for best results."""
                }),
                
                # Advanced Pitch Options
                "rvc_pitch_options": ("RVC_PITCH_OPTIONS", {
                    "tooltip": "Optional advanced pitch extraction settings from RVC Pitch Options node. Overrides basic parameters."
                }),
                
                "output_sample_rate": (sample_rates, {
                    "default": 0,
                    "tooltip": "Output sample rate (0=use input rate). 44100/48000 recommended for high quality"
                }),
                
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Processing device. Auto=optimal device detection"
                })
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    
    FUNCTION = "create_engine"
    
    DESCRIPTION = """
    RVC Engine - Real-time Voice Conversion
    
    ⚠️ Voice conversion only - does NOT generate speech from text.
    Will not work with TTS Text or TTS SRT nodes. Use with Voice Changer node only.
    
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
        pitch=0,
        index_ratio=0.75,
        consonant_protection=0.25,
        volume_envelope=0.25,
        hubert_model="auto: Automatically select best available model",
        rvc_pitch_options=None,
        output_sample_rate=0,
        device="auto"
    ):
        """
        Create RVC engine adapter with conversion parameters.
        Models are loaded separately via 🎭 Load RVC Character Model node.
        
        Returns:
            RVC engine adapter configured for voice conversion
        """
        try:
            # Create RVC adapter
            adapter = RVCEngineAdapter()
            
            # Parse HuBERT model selection (format: "key: description")
            hubert_key = hubert_model.split(": ")[0] if ": " in hubert_model else hubert_model
            
            # Ensure HuBERT model is available (download if needed)
            from engines.rvc.hubert_downloader import ensure_hubert_model
            hubert_path = ensure_hubert_model(hubert_key)
            
            if hubert_path:
                print(f"✅ HuBERT model ready: {hubert_key}")
            else:
                print(f"⚠️ Could not load HuBERT model {hubert_key}, RVC may use fallback")
            
            # Set up pitch parameters with sensible defaults
            final_pitch_params = {
                'pitch_shift': pitch,
                'f0_method': 'rmvpe',  # Default pitch detection method
                'index_rate': index_ratio,
                'protect': consonant_protection,
                'rms_mix_rate': volume_envelope,
                'resample_sr': output_sample_rate,
                'hubert_model': hubert_key,
                'hubert_path': hubert_path
            }
            
            if rvc_pitch_options:
                # Advanced pitch options override basic parameters
                if isinstance(rvc_pitch_options, dict):
                    final_pitch_params.update(rvc_pitch_options)
                    print("🔧 Using advanced pitch options from RVC Pitch Options node")
                else:
                    print("⚠️  Invalid pitch options format, using default rmvpe")
            else:
                print("🔧 Using default pitch detection: rmvpe")
            
            # Resolve device
            if device == "auto":
                import comfy.model_management as model_management
                device = str(model_management.get_torch_device())
            
            final_pitch_params['device'] = device
            
            # Store configuration in adapter (no models loaded here)
            adapter.config = {
                'type': 'rvc_engine',
                'engine_type': 'rvc',
                **final_pitch_params
            }
            
            print(f"⚙️ RVC Engine created - HuBERT: {hubert_key}, Pitch method: {final_pitch_params['f0_method']}, Device: {device}")
            if rvc_pitch_options:
                print("🔧 Advanced pitch options applied")
            
            return (adapter,)
        
        except Exception as e:
            print(f"❌ RVC Engine creation failed: {e}")
            # Return minimal adapter on failure
            try:
                adapter = RVCEngineAdapter()
                adapter.config = {
                    'type': 'rvc_engine',
                    'engine_type': 'rvc',
                    'error': str(e),
                    'device': 'cpu'
                }
                return (adapter,)
            except:
                return (None,)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for RVC engine creation."""
        return True