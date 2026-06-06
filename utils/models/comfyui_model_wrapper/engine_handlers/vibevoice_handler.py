"""
VibeVoice engine handler with special RAM management
"""

import torch
import gc
from typing import Optional, TYPE_CHECKING

from .generic_handler import GenericHandler
from ..cache_utils import invalidate_all_caches

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class VibeVoiceHandler(GenericHandler):
    """
    Handler for VibeVoice engine with special RAM management.
    
    VibeVoice requires complete deletion instead of CPU migration to prevent
    system RAM accumulation and memory leaks.
    """
    
    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        VibeVoice partial unload with RAM cleanup.
        
        Performs smart CPU migration with cleanup of old VibeVoice models
        to prevent system RAM accumulation.
        """
        if not wrapper._is_loaded_on_gpu:
            return 0

        model = wrapper._model_ref() if wrapper._model_ref else None
        dispatched_model = getattr(model, 'model', None) if model is not None else None

        if device == 'cpu' and getattr(dispatched_model, 'hf_device_map', None):
            print("🔄 VibeVoice/Kugel dispatched model cannot be CPU-migrated safely; invalidating for recreation")

            if model is not None:
                model.model = None
                model.processor = None

            wrapper.current_device = 'cpu'
            wrapper._is_loaded_on_gpu = False
            wrapper._is_valid_for_reuse = False

            invalidate_all_caches()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()

            return wrapper._memory_size
        
        # VibeVoice special handling: Smart CPU migration with RAM cleanup
        if device == 'cpu':
            print(f"🔄 VibeVoice: Smart CPU migration with RAM cleanup to prevent accumulation")
            
            # Before moving to CPU, clean up any existing VibeVoice models in system RAM  
            # Simple approach: clear other VibeVoice models that aren't on GPU
            try:
                models_cleared = 0
                # Import here to avoid circular imports
                from ..model_manager import tts_model_manager
                cache_keys_to_clear = []
                
                # Find VibeVoice models that are in CPU/RAM (not GPU loaded)
                for cache_key, cached_wrapper in tts_model_manager._model_cache.items():
                    if (cached_wrapper.model_info.engine == "vibevoice" and
                        not cached_wrapper._is_loaded_on_gpu and  # Model is in RAM/CPU
                        cached_wrapper != wrapper):  # Don't clear ourselves
                        cache_keys_to_clear.append(cache_key)
                        models_cleared += 1
                
                # Clear the old VibeVoice models from RAM
                for key in cache_keys_to_clear:
                    try:
                        tts_model_manager.remove_model(key)
                        print(f"🗑️ Removed VibeVoice model from RAM: {key[:16]}...")
                    except Exception as e:
                        print(f"⚠️ Failed to remove {key[:16]}: {e}")
                
                if models_cleared > 0:
                    print(f"🧹 RAM cleanup: removed {models_cleared} old VibeVoice models from system memory")
                    try:
                        gc.collect()
                    except Exception as gc_error:
                        print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")
                else:
                    print(f"🔍 No old VibeVoice models found in RAM")
                    
            except Exception as e:
                print(f"⚠️ RAM cleanup error: {e}")
            
            # Now proceed with normal CPU migration
            print(f"📥 Moving VibeVoice to CPU (RAM cleanup completed)")
        
        # Use standard CPU migration after cleanup
        return super().partially_unload(wrapper, device, memory_to_free)
    
    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """
        VibeVoice full unload using CPU migration (deletion doesn't actually free VRAM).
        """
        if memory_to_free is not None and memory_to_free < wrapper.loaded_size():
            # Try partial unload first
            freed = self.partially_unload(wrapper, 'cpu', memory_to_free)
            success = freed >= memory_to_free
            return success

        # Full unload - use CPU migration (proven to work)
        freed = self.partially_unload(wrapper, 'cpu', wrapper._memory_size)
        return freed > 0
