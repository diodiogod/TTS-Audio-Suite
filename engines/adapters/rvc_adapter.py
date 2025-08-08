"""
RVC Engine Adapter - Engine-specific adapter for RVC voice conversion
Provides standardized interface for RVC operations in voice changer system
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import sys
import os

# Add project root to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from engines.rvc.rvc_engine import RVCEngine


class RVCEngineAdapter:
    """Engine-specific adapter for RVC voice conversion."""
    
    def __init__(self, node_instance=None):
        """
        Initialize RVC adapter.
        
        Args:
            node_instance: RVC Engine node instance (optional)
        """
        self.node = node_instance
        self.engine_type = "rvc"
        self.rvc_engine = RVCEngine()
        self._loaded_models = {}
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get available RVC models and configurations.
        
        Returns:
            Dictionary with model information
        """
        return self.rvc_engine.get_available_models()
    
    def get_pitch_extraction_methods(self) -> list:
        """Get available pitch extraction methods for RVC."""
        return self.rvc_engine.get_pitch_extraction_methods()
    
    def load_models(self, rvc_model_path: str, hubert_model_path: str, 
                   index_path: Optional[str] = None) -> Dict[str, str]:
        """
        Load RVC and Hubert models for voice conversion.
        
        Args:
            rvc_model_path: Path to RVC voice model
            hubert_model_path: Path to Hubert feature extraction model
            index_path: Optional path to index file for enhanced quality
            
        Returns:
            Dictionary with model IDs
        """
        try:
            # Load models through RVC engine
            rvc_model_id = self.rvc_engine.load_rvc_model(rvc_model_path, index_path)
            hubert_model_id = self.rvc_engine.load_hubert_model(hubert_model_path)
            
            model_key = f"{rvc_model_id}_{hubert_model_id}"
            self._loaded_models[model_key] = {
                'rvc_id': rvc_model_id,
                'hubert_id': hubert_model_id,
                'rvc_path': rvc_model_path,
                'hubert_path': hubert_model_path,
                'index_path': index_path
            }
            
            return {
                'model_key': model_key,
                'rvc_model_id': rvc_model_id,
                'hubert_model_id': hubert_model_id
            }
            
        except Exception as e:
            print(f"Error loading RVC models: {e}")
            raise e
    
    def convert_voice(
        self,
        audio_input: Union[torch.Tensor, np.ndarray, tuple],
        model_key: str,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.25,
        f0_method: str = 'rmvpe',
        f0_autotune: bool = False,
        resample_sr: int = 0,
        crepe_hop_length: int = 160,
        use_cache: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Perform voice conversion using RVC.
        
        Args:
            audio_input: Input audio data
            model_key: Model identifier from load_models
            pitch_shift: Pitch shift in semitones (-14 to +14)
            index_rate: Index influence rate (0.0-1.0)
            rms_mix_rate: RMS mixing rate (0.0-1.0) 
            protect: Consonant protection (0.0-0.5)
            f0_method: Pitch extraction method
            f0_autotune: Enable autotune
            resample_sr: Resample rate (0 for no resampling)
            crepe_hop_length: Crepe hop length (16-512)
            use_cache: Whether to use caching
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (converted_audio, sample_rate)
        """
        try:
            # Get model IDs from loaded models
            if model_key not in self._loaded_models:
                raise ValueError(f"Model key {model_key} not found. Load models first.")
            
            model_info = self._loaded_models[model_key]
            rvc_model_id = model_info['rvc_id']
            hubert_model_id = model_info['hubert_id']
            
            # Prepare pitch extraction parameters
            pitch_params = {
                'f0_method': f0_method,
                'f0_autotune': f0_autotune,
                'index_rate': index_rate,
                'resample_sr': resample_sr,
                'rms_mix_rate': rms_mix_rate,
                'protect': protect,
                'crepe_hop_length': crepe_hop_length
            }
            
            # Perform voice conversion
            converted_audio, sample_rate = self.rvc_engine.convert_voice(
                audio=audio_input,
                rvc_model_id=rvc_model_id,
                hubert_model_id=hubert_model_id,
                pitch_shift=pitch_shift,
                pitch_params=pitch_params,
                use_cache=use_cache
            )
            
            return converted_audio, sample_rate
            
        except Exception as e:
            print(f"Error in RVC voice conversion: {e}")
            raise e
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Args:
            model_key: Model identifier
            
        Returns:
            Dictionary with model information
        """
        if model_key not in self._loaded_models:
            return {}
        
        model_info = self._loaded_models[model_key]
        return {
            'rvc_model': os.path.basename(model_info['rvc_path']),
            'hubert_model': os.path.basename(model_info['hubert_path']),
            'has_index': model_info['index_path'] is not None,
            'engine_type': self.engine_type
        }
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default RVC parameters."""
        return {
            'pitch_shift': 0,
            'index_rate': 0.75,
            'rms_mix_rate': 0.25,
            'protect': 0.25,
            'f0_method': 'rmvpe',
            'f0_autotune': False,
            'resample_sr': 0,
            'crepe_hop_length': 160
        }
    
    def validate_parameters(self, **params) -> Dict[str, Any]:
        """
        Validate and normalize RVC parameters.
        
        Returns:
            Validated parameters dictionary
        """
        validated = self.get_default_parameters()
        
        # Validate and update parameters
        if 'pitch_shift' in params:
            validated['pitch_shift'] = max(-14, min(14, int(params['pitch_shift'])))
        
        if 'index_rate' in params:
            validated['index_rate'] = max(0.0, min(1.0, float(params['index_rate'])))
        
        if 'rms_mix_rate' in params:
            validated['rms_mix_rate'] = max(0.0, min(1.0, float(params['rms_mix_rate'])))
        
        if 'protect' in params:
            validated['protect'] = max(0.0, min(0.5, float(params['protect'])))
        
        if 'f0_method' in params:
            available_methods = self.get_pitch_extraction_methods()
            if params['f0_method'] in available_methods:
                validated['f0_method'] = params['f0_method']
        
        if 'f0_autotune' in params:
            validated['f0_autotune'] = bool(params['f0_autotune'])
        
        if 'resample_sr' in params:
            valid_rates = [0, 16000, 32000, 40000, 44100, 48000]
            if params['resample_sr'] in valid_rates:
                validated['resample_sr'] = int(params['resample_sr'])
        
        if 'crepe_hop_length' in params:
            validated['crepe_hop_length'] = max(16, min(512, int(params['crepe_hop_length'])))
        
        return validated
    
    def cleanup(self):
        """Clean up loaded models and free memory."""
        self._loaded_models.clear()
        if hasattr(self, 'rvc_engine'):
            self.rvc_engine.cleanup()
        print("RVC adapter cleanup completed")