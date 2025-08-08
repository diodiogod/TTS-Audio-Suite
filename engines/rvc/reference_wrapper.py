"""
RVC Reference Implementation Wrapper
Direct wrapper around the working reference RVC code
"""

import os
import sys
import numpy as np
import torch
from typing import Tuple, Optional

# Add reference path to sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
reference_path = os.path.join(project_root, "docs", "RVC", "Comfy-RVC-For-Reference")

if reference_path not in sys.path:
    sys.path.insert(0, reference_path)

class RVCReferenceWrapper:
    """
    Wrapper around the working reference RVC implementation
    Provides clean interface for TTS Suite integration
    """
    
    def __init__(self):
        self.hubert_model = None
        self.model_cache = {}
        
    def load_model(self, model_path: str, index_path: Optional[str] = None):
        """Load RVC model using reference implementation"""
        try:
            # Import reference functions
            from vc_infer_pipeline import get_vc
            
            print(f"🔄 Loading RVC model via reference wrapper: {os.path.basename(model_path)}")
            
            # Use reference implementation to load model
            model_data = get_vc(model_path, index_path)
            
            if model_data:
                print(f"✅ Reference RVC model loaded: {model_data['model_name']}")
                return model_data
            else:
                print("❌ Reference RVC model loading failed")
                return None
                
        except Exception as e:
            print(f"❌ Reference wrapper error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_hubert(self, hubert_path: str):
        """Load Hubert model using reference implementation"""
        try:
            # Import reference functions
            from lib.model_utils import load_hubert
            from config import config
            
            print(f"🔄 Loading Hubert model via reference: {os.path.basename(hubert_path)}")
            
            # Use reference implementation
            self.hubert_model = load_hubert(hubert_path, config)
            
            if self.hubert_model:
                print("✅ Reference Hubert model loaded")
                return self.hubert_model
            else:
                print("❌ Reference Hubert model loading failed")
                return None
                
        except Exception as e:
            print(f"❌ Reference Hubert loading error: {e}")
            return None
    
    def convert_voice(self, 
                     audio: np.ndarray, 
                     sample_rate: int,
                     model_data: dict,
                     f0_up_key: int = 0,
                     f0_method: str = "rmvpe",
                     index_rate: float = 0.75,
                     protect: float = 0.33,
                     rms_mix_rate: float = 0.25,
                     **kwargs) -> Optional[Tuple[np.ndarray, int]]:
        """
        Perform voice conversion using reference implementation
        """
        try:
            # Import reference functions
            from vc_infer_pipeline import vc_single
            
            print(f"🎵 Reference RVC conversion: {f0_method} method, pitch: {f0_up_key}")
            
            # Ensure we have required components
            if not (model_data and self.hubert_model):
                print("❌ Missing model components for reference conversion")
                return None
            
            # Prepare input audio in reference format
            input_audio = (audio, sample_rate)
            
            # Call reference vc_single function with all components
            result = vc_single(
                cpt=model_data["cpt"],
                net_g=model_data["net_g"],
                vc=model_data["vc"],
                hubert_model=self.hubert_model,
                sid=0,  # speaker id
                input_audio=input_audio,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                file_index=model_data["file_index"],
                index_rate=index_rate,
                protect=protect,
                rms_mix_rate=rms_mix_rate,
                **kwargs
            )
            
            if result:
                output_audio, output_sr = result
                print(f"✅ Reference RVC conversion completed")
                return (output_audio, output_sr)
            else:
                print("❌ Reference RVC conversion failed")
                return None
                
        except Exception as e:
            print(f"❌ Reference conversion error: {e}")
            import traceback
            traceback.print_exc()
            return None

# Global wrapper instance
reference_wrapper = RVCReferenceWrapper()