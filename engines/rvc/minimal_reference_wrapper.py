"""
Minimal RVC Reference Implementation Wrapper
Calls the original reference code directly with minimal modifications
"""

import os
import sys
import numpy as np
import torch
from typing import Tuple, Optional

class MinimalRVCWrapper:
    """
    Minimal wrapper that directly calls the working reference implementation
    Uses direct imports from the reference directory without copying code
    """
    
    def __init__(self):
        self.hubert_model = None
        self.reference_path = None
        self._setup_reference_path()
        
    def _setup_reference_path(self):
        """Setup path to implementation (moved from docs to proper engine location)"""
        current_dir = os.path.dirname(__file__)
        self.reference_path = os.path.join(current_dir, "impl")
        self.lib_path = os.path.join(self.reference_path, "lib")
        
        # Add all necessary paths for reference implementation
        self.infer_pack_path = os.path.join(self.lib_path, "infer_pack")
        self.text_path = os.path.join(self.infer_pack_path, "text")
        
        # Add paths in order of priority
        if self.text_path not in sys.path:
            sys.path.insert(0, self.text_path)       # For symbols
        if self.infer_pack_path not in sys.path:
            sys.path.insert(0, self.infer_pack_path)  # For modules, attentions, commons
        if self.lib_path not in sys.path:
            sys.path.insert(0, self.lib_path)        # For infer_pack, utils, etc.
        if self.reference_path not in sys.path:
            sys.path.insert(0, self.reference_path)  # For config, vc_infer_pipeline, etc.
    
    def convert_voice(self, 
                     audio: np.ndarray, 
                     sample_rate: int,
                     model_path: str,
                     index_path: Optional[str] = None,
                     f0_up_key: int = 0,
                     f0_method: str = "rmvpe",
                     index_rate: float = 0.75,
                     protect: float = 0.33,
                     rms_mix_rate: float = 0.25,
                     **kwargs) -> Optional[Tuple[np.ndarray, int]]:
        """
        Perform voice conversion using direct reference calls
        """
        try:
            print(f"🎵 Minimal wrapper RVC conversion: {f0_method} method, pitch: {f0_up_key}")
            
            # Import reference functions only when needed (lazy import)
            # This avoids the import issues at module load time
            with self._temporary_cwd():
                from vc_infer_pipeline import get_vc, vc_single
                from lib.model_utils import load_hubert
                from config import config
                
                # Load RVC model
                print(f"🔄 Loading RVC model via minimal wrapper: {os.path.basename(model_path)}")
                model_data = get_vc(model_path, index_path)
                
                if not model_data:
                    print("❌ Failed to load RVC model")
                    return None
                
                # Load Hubert model
                hubert_path = self._find_hubert_model()
                if not hubert_path:
                    print("❌ Hubert model not found")
                    return None
                
                print(f"🔄 Loading Hubert model: {os.path.basename(hubert_path)}")
                hubert_model = load_hubert(hubert_path, config)
                if not hubert_model:
                    print("❌ Failed to load Hubert model")
                    return None
                
                # Prepare input audio
                input_audio = (audio, sample_rate)
                
                # Ensure RMVPE model is available for reference implementation
                if f0_method in ["rmvpe", "rmvpe+", "rmvpe_onnx"]:
                    from utils.downloads.model_downloader import download_rmvpe_for_reference
                    rmvpe_path = download_rmvpe_for_reference()
                    if not rmvpe_path:
                        print("⚠️ RMVPE model not available, continuing anyway...")
                
                # Call reference vc_single function
                result = vc_single(
                    cpt=model_data["cpt"],
                    net_g=model_data["net_g"],
                    vc=model_data["vc"],
                    hubert_model=hubert_model,
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
                    print(f"✅ Minimal wrapper RVC conversion completed")
                    return (output_audio, output_sr)
                else:
                    print("❌ RVC conversion returned None")
                    return None
                
        except Exception as e:
            print(f"❌ Minimal wrapper conversion error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _temporary_cwd(self):
        """Context manager to temporarily change working directory for imports"""
        class TempCWD:
            def __init__(self, path):
                self.path = path
                self.old_cwd = None
                
            def __enter__(self):
                self.old_cwd = os.getcwd()
                os.chdir(self.path)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                os.chdir(self.old_cwd)
        
        return TempCWD(self.reference_path)
    
    def _find_hubert_model(self) -> Optional[str]:
        """Find available Hubert model."""
        try:
            import folder_paths
            models_dir = folder_paths.models_dir
            
            # Common Hubert model names and locations
            hubert_candidates = [
                "content-vec-best.safetensors",
                "hubert_base.pt",
                "chinese-hubert-base.pt",
                "chinese-wav2vec2-base.pt"
            ]
            
            for model_name in hubert_candidates:
                model_path = os.path.join(models_dir, model_name)
                if os.path.exists(model_path):
                    print(f"📄 Found Hubert model: {model_name}")
                    return model_path
            
            print("❌ No Hubert model found")
            return None
            
        except Exception as e:
            print(f"Error finding Hubert model: {e}")
            return None

# Global wrapper instance
minimal_wrapper = MinimalRVCWrapper()