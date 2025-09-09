"""
Minimal RVC Reference Implementation Wrapper
Calls the original reference code directly with minimal modifications
"""

import os
import sys

# CRITICAL FIX for Python 3.13 + numba + librosa compatibility
# 🔬 NUMBA WORKAROUND: Commented out - testing if still needed with numba 0.61.2+ and librosa 0.11.0+
# Only disable numba JIT on Python 3.13+ where it causes compatibility issues
# if sys.version_info >= (3, 13):
#     os.environ['NUMBA_DISABLE_JIT'] = '1'
#     print("🔧 RVC: Disabled numba JIT for Python 3.13+ compatibility")

# Additional librosa compatibility monkey-patching (keeping this active since it's useful)
def apply_librosa_compatibility_patches():
    """Apply global librosa compatibility patches for Python 3.13"""
    try:
        import librosa.util
        import numpy as np
        
        # Check if pad_center is missing and add it
        if not hasattr(librosa.util, 'pad_center'):
            def pad_center(data, size, axis=-1, **kwargs):
                """Manual implementation of librosa's pad_center for compatibility"""
                n = data.shape[axis]
                lpad = int((size - n) // 2)
                rpad = int(size - n - lpad)
                pad_widths = [(0, 0)] * data.ndim
                pad_widths[axis] = (lpad, rpad)
                return np.pad(data, pad_widths, mode=kwargs.get('mode', 'constant'), 
                             constant_values=kwargs.get('constant_values', 0))
            
            librosa.util.pad_center = pad_center
            print("🔧 RVC: Applied pad_center compatibility patch to librosa.util")
        
        # Check if tiny is missing and add it  
        if not hasattr(librosa.util, 'tiny'):
            def tiny(x):
                """Manual implementation of librosa's tiny function for compatibility"""
                return np.finfo(np.float32).tiny
            librosa.util.tiny = tiny
            print("🔧 RVC: Applied tiny compatibility patch to librosa.util")
        
        # Check if fill_off_diagonal is missing and add it
        if not hasattr(librosa.util, 'fill_off_diagonal'):
            def fill_off_diagonal(x, radius, value=0):
                """Manual implementation of librosa's fill_off_diagonal for compatibility"""
                x = np.asarray(x)
                if x.ndim == 1:
                    # For 1D arrays, return a copy
                    return x.copy()
                
                # Create a copy to avoid modifying the original
                result = x.copy()
                n = min(result.shape)
                
                # Fill off-diagonal elements within the specified radius
                for i in range(n):
                    for j in range(n):
                        if abs(i - j) <= radius and i != j:
                            result[i, j] = value
                
                return result
            librosa.util.fill_off_diagonal = fill_off_diagonal
            print("🔧 RVC: Applied fill_off_diagonal compatibility patch to librosa.util")
        
        # Check if is_positive_int is missing and add it
        if not hasattr(librosa.util, 'is_positive_int'):
            def is_positive_int(x):
                """Manual implementation of librosa's is_positive_int for compatibility"""
                try:
                    return isinstance(x, int) and x > 0
                except (TypeError, ValueError):
                    return False
            librosa.util.is_positive_int = is_positive_int
            print("🔧 RVC: Applied is_positive_int compatibility patch to librosa.util")
        
        # Check if expand_to is missing and add it
        if not hasattr(librosa.util, 'expand_to'):
            def expand_to(x, *, ndim=None, axes=None):
                """Manual implementation of librosa's expand_to for compatibility"""
                x = np.asarray(x)
                
                if ndim is not None:
                    # Expand to target number of dimensions
                    while x.ndim < ndim:
                        x = np.expand_dims(x, axis=-1)
                
                if axes is not None:
                    # Expand along specific axes
                    for axis in sorted(axes):
                        if axis >= x.ndim:
                            x = np.expand_dims(x, axis=axis)
                
                return x
            librosa.util.expand_to = expand_to
            print("🔧 RVC: Applied expand_to compatibility patch to librosa.util")
            
    except Exception as e:
        print(f"⚠️ RVC: Could not apply librosa compatibility patches: {e}")
    
    # Apply patches after a short delay to ensure librosa is loaded
    import threading
    threading.Timer(0.1, apply_librosa_compatibility_patches).start()

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
            
            # Apply librosa patches before importing RVC modules
            if sys.version_info >= (3, 13):
                try:
                    import librosa.util
                    if not hasattr(librosa.util, 'pad_center'):
                        def pad_center(data, size, axis=-1, **kwargs):
                            """Manual implementation of librosa's pad_center for compatibility"""
                            import numpy as np
                            n = data.shape[axis]
                            lpad = int((size - n) // 2)
                            rpad = int(size - n - lpad)
                            pad_widths = [(0, 0)] * data.ndim
                            pad_widths[axis] = (lpad, rpad)
                            return np.pad(data, pad_widths, mode=kwargs.get('mode', 'constant'), 
                                         constant_values=kwargs.get('constant_values', 0))
                        librosa.util.pad_center = pad_center
                        print("🔧 RVC: Applied pad_center compatibility patch")
                    
                    if not hasattr(librosa.util, 'tiny'):
                        def tiny(x):
                            """Manual implementation of librosa's tiny function for compatibility"""
                            import numpy as np
                            return np.finfo(np.float32).tiny
                        librosa.util.tiny = tiny
                        print("🔧 RVC: Applied tiny compatibility patch")
                    
                    if not hasattr(librosa.util, 'fill_off_diagonal'):
                        def fill_off_diagonal(x, radius, value=0):
                            """Manual implementation of librosa's fill_off_diagonal for compatibility"""
                            import numpy as np
                            x = np.asarray(x)
                            if x.ndim == 1:
                                # For 1D arrays, return a copy
                                return x.copy()
                            
                            # Create a copy to avoid modifying the original
                            result = x.copy()
                            n = min(result.shape)
                            
                            # Fill off-diagonal elements within the specified radius
                            for i in range(n):
                                for j in range(n):
                                    if abs(i - j) <= radius and i != j:
                                        result[i, j] = value
                            
                            return result
                        librosa.util.fill_off_diagonal = fill_off_diagonal
                        print("🔧 RVC: Applied fill_off_diagonal compatibility patch")
                    
                    if not hasattr(librosa.util, 'is_positive_int'):
                        def is_positive_int(x):
                            """Manual implementation of librosa's is_positive_int for compatibility"""
                            try:
                                return isinstance(x, int) and x > 0
                            except (TypeError, ValueError):
                                return False
                        librosa.util.is_positive_int = is_positive_int
                        print("🔧 RVC: Applied is_positive_int compatibility patch")
                    
                    if not hasattr(librosa.util, 'expand_to'):
                        def expand_to(x, *, ndim=None, axes=None):
                            """Manual implementation of librosa's expand_to for compatibility"""
                            import numpy as np
                            x = np.asarray(x)
                            
                            if ndim is not None:
                                # Expand to target number of dimensions
                                while x.ndim < ndim:
                                    x = np.expand_dims(x, axis=-1)
                            
                            if axes is not None:
                                # Expand along specific axes - handle both int and iterable
                                axes_list = [axes] if isinstance(axes, int) else axes
                                for axis in sorted(axes_list):
                                    if axis >= x.ndim:
                                        x = np.expand_dims(x, axis=axis)
                            
                            return x
                        librosa.util.expand_to = expand_to
                        print("🔧 RVC: Applied expand_to compatibility patch")
                        
                except ImportError:
                    pass  # librosa not loaded yet
            
            # Import reference functions using absolute imports to avoid package issues
            from engines.rvc.impl.vc_infer_pipeline import get_vc, vc_single
            from engines.rvc.impl.lib.model_utils import load_hubert
            from engines.rvc.impl.config import config
            
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
                print(f"🔧 RVC: {f0_method} method requires RMVPE model, checking availability...")
                try:
                    from utils.downloads.model_downloader import download_rmvpe_for_reference
                    rmvpe_path = download_rmvpe_for_reference()
                    if rmvpe_path:
                        print(f"✅ RMVPE model ready at: {rmvpe_path}")
                    else:
                        print("❌ RMVPE model download failed, RVC may fail with this f0_method")
                except Exception as e:
                    print(f"❌ RMVPE download error: {e}")
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
            
            # Common Hubert model names and locations - RVC compatible first
            hubert_candidates = [
                "content-vec-best.safetensors",  # RVC library expects this specifically
                "hubert_base.pt",
                "chinese-hubert-base.pt",
                "hubert_base_jp.pt",
                "hubert_base_kr.pt",
                "chinese-wav2vec2-base.pt"
            ]
            
            for model_name in hubert_candidates:
                # Try TTS path first, then legacy locations
                search_paths = [
                    os.path.join(models_dir, "TTS", "hubert", model_name),
                    os.path.join(models_dir, "TTS", model_name),
                    os.path.join(models_dir, "hubert", model_name),  # Legacy
                    os.path.join(models_dir, model_name)  # Legacy - direct in models/
                ]
                
                for model_path in search_paths:
                    if os.path.exists(model_path):
                        print(f"📄 Found Hubert model: {model_name} at {model_path}")
                        return model_path
            
            # If no model found, try to download content-vec-best as fallback
            print("❌ No compatible Hubert model found locally")
            print("📥 Attempting to download RVC-compatible model as fallback...")
            
            try:
                from engines.rvc.hubert_downloader import find_or_download_hubert
                fallback_path = find_or_download_hubert("content-vec-best", models_dir)
                if fallback_path:
                    print(f"✅ Downloaded RVC-compatible fallback: {fallback_path}")
                    return fallback_path
                else:
                    print("❌ Failed to download fallback model")
            except Exception as e:
                print(f"❌ Fallback download failed: {e}")
            
            return None
            
        except Exception as e:
            print(f"Error finding Hubert model: {e}")
            return None

# Global wrapper instance
minimal_wrapper = MinimalRVCWrapper()