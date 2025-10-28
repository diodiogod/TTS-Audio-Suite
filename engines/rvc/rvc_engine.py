"""
RVC Engine - Core RVC voice conversion implementation for TTS Audio Suite
Consolidates functionality from reference RVC nodes into unified engine
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import tempfile
import hashlib
from pathlib import Path

# Import audio processing utilities from the project
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.processing import AudioProcessingUtils
import comfy.model_management as model_management

# Import unified model interface for ComfyUI integration
from utils.models.unified_model_interface import load_vc_model, load_auxiliary_model

class RVCModelWrapper:
    """
    Wrapper for RVC model dict to provide .to() method for device management.

    RVC models are loaded as dicts with 'net_g', 'cpt', 'vc', etc.
    This wrapper makes them compatible with ComfyUI's model management.
    """
    def __init__(self, model_data: dict, device: str):
        self.model_data = model_data
        self.device = device
        # ComfyUI expects .model.model attribute for logging - point to net_g
        self.model = model_data.get('net_g', model_data)

    def to(self, device):
        """Move all model components to target device"""
        self.device = device

        # Move net_g (the main synthesizer network)
        if 'net_g' in self.model_data and hasattr(self.model_data['net_g'], 'to'):
            self.model_data['net_g'] = self.model_data['net_g'].to(device)
            self.model = self.model_data['net_g']  # Update reference

        # Move VC pipeline if it has components
        if 'vc' in self.model_data:
            vc = self.model_data['vc']
            if hasattr(vc, 'to'):
                self.model_data['vc'] = vc.to(device)

        return self

    def __getitem__(self, key):
        """Allow dict-like access"""
        return self.model_data[key]

    def get(self, key, default=None):
        """Allow dict-like get"""
        return self.model_data.get(key, default)


class RVCEngine:
    """
    Core RVC (Real-time Voice Conversion) Engine
    Consolidates model loading, pitch extraction, and voice conversion into unified interface
    """
    
    def __init__(self):
        self.device = model_management.get_torch_device()
        self.rvc_models = {}
        self.hubert_models = {}
        self.pitch_extractors = {}
        self.cache_dir = None
        self._setup_cache()
        
        # Default pitch extraction options
        self.default_pitch_params = {
            'f0_method': 'rmvpe',
            'f0_autotune': False,
            'index_rate': 0.75,
            'resample_sr': 0,
            'rms_mix_rate': 0.25,
            'protect': 0.25,
            'crepe_hop_length': 160
        }

    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.
        """
        self.device = device

        # Move all loaded RVC models
        for model_name, model in self.rvc_models.items():
            if hasattr(model, 'to'):
                self.rvc_models[model_name] = model.to(device)

        # Move all loaded Hubert models
        for model_name, model in self.hubert_models.items():
            if hasattr(model, 'to'):
                self.hubert_models[model_name] = model.to(device)

        # Move all pitch extractors
        for extractor_name, extractor in self.pitch_extractors.items():
            if hasattr(extractor, 'to'):
                self.pitch_extractors[extractor_name] = extractor.to(device)

        return self

    def _setup_cache(self):
        """Setup cache directory for RVC processing"""
        try:
            import folder_paths
            temp_path = folder_paths.get_temp_directory()
            self.cache_dir = os.path.join(temp_path, "rvc_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
        except ImportError:
            # Fallback if folder_paths not available
            self.cache_dir = os.path.join(tempfile.gettempdir(), "rvc_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_available_models(self) -> Dict[str, list]:
        """
        Get available RVC and Hubert models
        Returns dict with 'rvc_models' and 'hubert_models' lists
        """
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
            
            rvc_models = []
            hubert_models = []
            
            # Look for RVC models - try TTS path first, then legacy
            rvc_search_paths = [
                os.path.join(models_dir, "TTS", "RVC"),
                os.path.join(models_dir, "RVC")  # Legacy
            ]
            
            for rvc_dir in rvc_search_paths:
                if os.path.exists(rvc_dir):
                    for file in os.listdir(rvc_dir):
                        if file.endswith('.pth'):
                            relative_path = os.path.relpath(os.path.join(rvc_dir, file), models_dir)
                            rvc_models.append(relative_path)
            
            # Look for Hubert models (common names) - try TTS path first, then legacy
            common_hubert_models = [
                "content-vec-best.safetensors",
                "hubert-base.pt", 
                "chinese-hubert-base.pt"
            ]
            
            hubert_search_paths = [
                os.path.join(models_dir, "TTS"),
                os.path.join(models_dir)  # Legacy - direct in models/
            ]
            
            for search_dir in hubert_search_paths:
                for model_file in common_hubert_models:
                    full_path = os.path.join(search_dir, model_file)
                    if os.path.exists(full_path):
                        relative_path = os.path.relpath(full_path, models_dir)
                        if relative_path not in hubert_models:
                            hubert_models.append(relative_path)
            
            # Look for any .pt or .safetensors files that might be Hubert models
            for file in os.listdir(models_dir):
                if file.endswith(('.pt', '.safetensors')) and 'hubert' in file.lower():
                    if file not in hubert_models:
                        hubert_models.append(file)
            
            return {
                'rvc_models': rvc_models if rvc_models else ["No RVC models found"],
                'hubert_models': hubert_models if hubert_models else ["content-vec-best.safetensors"]
            }
            
        except Exception as e:
            print(f"Error getting available models: {e}")
            return {
                'rvc_models': ["No RVC models found"],
                'hubert_models': ["content-vec-best.safetensors"]
            }
    
    def get_pitch_extraction_methods(self) -> list:
        """Get available pitch extraction methods"""
        return [
            'rmvpe',
            'rmvpe+', 
            'mangio-crepe',
            'crepe',
            'pm',
            'harvest',
            'dio',
            'fcpe'
        ]
    
    def load_rvc_model(self, model_path: str, index_path: Optional[str] = None) -> str:
        """
        Load RVC model for voice conversion
        
        Args:
            model_path: Path to RVC .pth model file
            index_path: Optional path to .index file for enhanced quality
            
        Returns:
            Model identifier for later use
        """
        try:
            # Create unique model identifier
            model_id = hashlib.md5(f"{model_path}_{index_path or ''}".encode()).hexdigest()
            
            if model_id not in self.rvc_models:
                # Store model configuration (actual loading would happen during inference)
                self.rvc_models[model_id] = {
                    'model_path': model_path,
                    'index_path': index_path,
                    'loaded': False,
                    'model_obj': None
                }
                
                print(f"RVC model registered: {os.path.basename(model_path)}")
            
            return model_id
            
        except Exception as e:
            print(f"Error loading RVC model: {e}")
            raise e
    
    def load_hubert_model(self, model_path: str) -> str:
        """
        Load Hubert feature extraction model
        
        Args:
            model_path: Path to Hubert model file
            
        Returns:
            Model identifier for later use
        """
        try:
            model_id = hashlib.md5(model_path.encode()).hexdigest()
            
            if model_id not in self.hubert_models:
                # Store model configuration (actual loading would happen during inference)
                self.hubert_models[model_id] = {
                    'model_path': model_path,
                    'loaded': False,
                    'model_obj': None
                }
                
                print(f"Hubert model registered: {os.path.basename(model_path)}")
            
            return model_id
            
        except Exception as e:
            print(f"Error loading Hubert model: {e}")
            raise e
    
    def convert_voice(
        self,
        audio: Union[torch.Tensor, np.ndarray, tuple],
        rvc_model_id: str,
        hubert_model_id: str,
        pitch_shift: int = 0,
        pitch_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Perform voice conversion using RVC

        Args:
            audio: Input audio (tensor, numpy array, or tuple of (audio, sample_rate))
            rvc_model_id: RVC model identifier from load_rvc_model
            hubert_model_id: Hubert model identifier from load_hubert_model
            pitch_shift: Pitch shift in semitones (-14 to +14)
            pitch_params: Pitch extraction parameters (optional)
            use_cache: Whether to use caching for faster repeated conversions

        Returns:
            Tuple of (converted_audio, sample_rate)
        """
        # CRITICAL FIX: Reload models to correct device if they were offloaded
        # Check if models need to be moved back to GPU after "Clear VRAM"
        import torch
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check RVC models
        for model_name, model in self.rvc_models.items():
            if hasattr(model, 'parameters'):
                try:
                    first_param = next(model.parameters())
                    current_device = str(first_param.device)
                    if current_device != str(target_device):
                        print(f"🔄 Reloading RVC model '{model_name}' from {current_device} to {target_device}")
                        self.to(target_device)
                        break
                except StopIteration:
                    pass

        # Check Hubert models
        for model_name, model in self.hubert_models.items():
            if hasattr(model, 'parameters'):
                try:
                    first_param = next(model.parameters())
                    current_device = str(first_param.device)
                    if current_device != str(target_device):
                        print(f"🔄 Reloading Hubert model '{model_name}' from {current_device} to {target_device}")
                        self.to(target_device)
                        break
                except StopIteration:
                    pass

        try:
            # Process input audio to consistent format
            if isinstance(audio, tuple):
                audio_data, sample_rate = audio
            else:
                # If audio is tensor/array without sample rate, assume common rate
                audio_data = audio
                sample_rate = 22050  # Default sample rate
            
            # Convert to numpy array if tensor
            if hasattr(audio_data, 'numpy'):
                audio_data = audio_data.numpy()
            elif isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().cpu().numpy()
                
            # Ensure audio is 1D
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=0) if audio_data.shape[0] < audio_data.shape[1] else audio_data.mean(axis=1)
            
            # Merge pitch parameters with defaults
            final_pitch_params = self.default_pitch_params.copy()
            if pitch_params:
                final_pitch_params.update(pitch_params)
            
            # Check cache first
            cache_key = self._get_cache_key(audio_data, rvc_model_id, hubert_model_id, pitch_shift, final_pitch_params)
            if use_cache:
                cached_result = self._get_cached_result(cache_key)
                if cached_result is not None:
                    print("Using cached RVC conversion")
                    return cached_result
            
            # Load models if not already loaded
            rvc_model_data = self._load_rvc_model_data(rvc_model_id)
            hubert_model_data = self._load_hubert_model_data(hubert_model_id)
            
            if not rvc_model_data or not hubert_model_data:
                print("⚠️ RVC models not available, returning original audio")
                return audio_data, sample_rate
            
            # Perform actual RVC conversion using the inference pipeline
            from .impl.vc_infer_pipeline import vc_single
            
            print(f"🔄 RVC Voice Conversion - Model: {rvc_model_data.get('model_name', 'Unknown')}")
            print(f"Pitch: {pitch_shift} semitones, Method: {final_pitch_params['f0_method']}")
            
            # Run voice conversion with correct parameters for vc_single function
            result = vc_single(
                cpt=rvc_model_data.get('cpt'),
                net_g=rvc_model_data.get('net_g'),
                vc=rvc_model_data.get('vc'),
                hubert_model=hubert_model_data,
                sid=0,  # Speaker ID - default to 0
                input_audio=(audio_data, sample_rate),
                f0_up_key=pitch_shift,
                f0_method=final_pitch_params['f0_method'],
                index_rate=final_pitch_params['index_rate'],
                protect=final_pitch_params['protect'],
                rms_mix_rate=final_pitch_params['rms_mix_rate'],
                resample_sr=final_pitch_params['resample_sr'],
                crepe_hop_length=final_pitch_params.get('crepe_hop_length', 160),
                f0_autotune=final_pitch_params.get('f0_autotune', False),
                file_index=rvc_model_data.get('file_index', '')
            )
            
            if result is None:
                print("⚠️ RVC conversion failed, returning original audio")
                return audio_data, sample_rate
                
            processed_audio, output_sr = result
            
            # Convert int16 to float32 if needed
            if processed_audio.dtype == np.int16:
                processed_audio = processed_audio.astype(np.float32) / 32768.0
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, (processed_audio, output_sr))
            
            print("✅ RVC voice conversion completed successfully")
            return processed_audio, output_sr
            
        except Exception as e:
            print(f"❌ Error in RVC voice conversion: {e}")
            print("🔄 Returning original audio")
            # Return original audio on error
            return audio_data, sample_rate
    
    def _load_rvc_model_data(self, rvc_model_id: str):
        """Load RVC model data for inference using ComfyUI model management"""
        try:
            if rvc_model_id not in self.rvc_models:
                return None
                
            model_info = self.rvc_models[rvc_model_id]
            
            # Load model if not already loaded
            if not model_info.get('loaded', False):
                model_path = model_info['model_path']
                index_path = model_info.get('index_path')
                
                print(f"🔄 Loading RVC model: {os.path.basename(model_path)}")
                
                try:
                    # Use unified model interface for ComfyUI integration
                    model_data = load_vc_model(
                        engine_name="rvc",
                        model_name=os.path.basename(model_path),
                        device=self.device,
                        model_path=model_path,
                        index_path=index_path
                    )
                    
                    if model_data:
                        model_info['model_obj'] = model_data
                        model_info['loaded'] = True
                        print(f"✅ RVC model loaded successfully (ComfyUI managed)")
                    else:
                        raise Exception("Model loading returned None")
                        
                except Exception as e:
                    print(f"⚠️ Failed to load via unified interface: {e}")
                    print(f"🔄 Falling back to direct loading...")
                    
                    # Fallback to direct loading using actual RVC implementation
                    from .impl.vc_infer_pipeline import get_vc
                    from .impl.config import config as rvc_config
                    
                    model_data = get_vc(model_path, index_path, rvc_config, self.device)
                    
                    if model_data:
                        model_info['model_obj'] = model_data
                        model_info['loaded'] = True
                        print(f"✅ RVC model loaded successfully (fallback)")
                else:
                    print(f"❌ Failed to load RVC model")
                    return None
            
            return model_info.get('model_obj')
            
        except Exception as e:
            print(f"Error loading RVC model data: {e}")
            return None
    
    def _load_hubert_model_data(self, hubert_model_id: str):
        """Load Hubert model data for inference"""
        try:
            if hubert_model_id not in self.hubert_models:
                return None
                
            model_info = self.hubert_models[hubert_model_id]
            
            # Load model if not already loaded
            if not model_info.get('loaded', False):
                model_path = model_info['model_path']
                
                print(f"🔄 Loading Hubert model: {os.path.basename(model_path)}")
                
                try:
                    # Use unified model interface for ComfyUI integration
                    model_obj = load_auxiliary_model(
                        engine_name="rvc",
                        model_type="hubert",
                        model_name=os.path.basename(model_path),
                        device=self.device,
                        model_path=model_path
                    )
                    
                    if model_obj:
                        model_info['model_obj'] = model_obj
                        model_info['loaded'] = True
                        print(f"✅ Hubert model loaded successfully (ComfyUI managed)")
                    else:
                        raise Exception("Model loading returned None")
                        
                except Exception as e:
                    print(f"⚠️ Failed to load via unified interface: {e}")
                    print(f"🔄 Falling back to direct loading...")
                    
                    # Fallback to direct loading using actual RVC implementation
                    from .impl.lib.model_utils import load_hubert
                    from .impl.config import config as rvc_config
                    
                    model_obj = load_hubert(model_path, rvc_config)
                    
                    if model_obj:
                        model_info['model_obj'] = model_obj
                        model_info['loaded'] = True
                        print(f"✅ Hubert model loaded successfully (fallback)")
                    else:
                        print(f"❌ Failed to load Hubert model")
                        return None
            
            return model_info.get('model_obj')
            
        except Exception as e:
            print(f"Error loading Hubert model data: {e}")
            return None
    
    def _get_cache_key(self, audio_data: np.ndarray, rvc_model_id: str, hubert_model_id: str, 
                      pitch_shift: int, pitch_params: Dict[str, Any]) -> str:
        """Generate cache key for conversion parameters"""
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()[:16]
        param_str = f"{rvc_model_id}_{hubert_model_id}_{pitch_shift}_{str(sorted(pitch_params.items()))}"
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        return f"{audio_hash}_{param_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Tuple[np.ndarray, int]]:
        """Retrieve cached conversion result"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
            if os.path.exists(cache_file):
                cached_data = np.load(cache_file, allow_pickle=True).item()
                return cached_data['audio'], cached_data['sample_rate']
            return None
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: Tuple[np.ndarray, int]):
        """Cache conversion result"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
            cache_data = {
                'audio': result[0],
                'sample_rate': result[1]
            }
            np.save(cache_file, cache_data)
        except Exception as e:
            print(f"Warning: Failed to cache RVC result: {e}")
    
    def cleanup(self):
        """Clean up loaded models and free memory"""
        for model_info in self.rvc_models.values():
            if model_info.get('model_obj'):
                del model_info['model_obj']
                model_info['loaded'] = False
                
        for model_info in self.hubert_models.values():
            if model_info.get('model_obj'):
                del model_info['model_obj']
                model_info['loaded'] = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("RVC Engine cleanup completed")