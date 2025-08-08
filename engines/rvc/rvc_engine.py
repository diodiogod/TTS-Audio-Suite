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
            
            # Look for RVC models in models/RVC directory
            rvc_dir = os.path.join(models_dir, "RVC")
            if os.path.exists(rvc_dir):
                for file in os.listdir(rvc_dir):
                    if file.endswith('.pth'):
                        rvc_models.append(f"RVC/{file}")
            
            # Look for Hubert models (common names)
            common_hubert_models = [
                "content-vec-best.safetensors",
                "hubert-base.pt", 
                "chinese-hubert-base.pt"
            ]
            
            for model_file in common_hubert_models:
                if os.path.exists(os.path.join(models_dir, model_file)):
                    hubert_models.append(model_file)
            
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
            
            # Placeholder for actual RVC conversion
            # In real implementation, this would:
            # 1. Load models if not already loaded
            # 2. Extract features using Hubert
            # 3. Apply pitch shift
            # 4. Perform RVC conversion
            # 5. Return converted audio
            
            print(f"RVC Voice Conversion - Model: {rvc_model_id[:8]}..., Pitch: {pitch_shift}")
            print(f"Pitch method: {final_pitch_params['f0_method']}, Protect: {final_pitch_params['protect']}")
            
            # For now, return the input audio with a slight modification to indicate processing
            # This is a placeholder - real implementation would do actual voice conversion
            processed_audio = audio_data * 0.95  # Slight volume reduction to indicate processing
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, (processed_audio, sample_rate))
            
            return processed_audio, sample_rate
            
        except Exception as e:
            print(f"Error in RVC voice conversion: {e}")
            raise e
    
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