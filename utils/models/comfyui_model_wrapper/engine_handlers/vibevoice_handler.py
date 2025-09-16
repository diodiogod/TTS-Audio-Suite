"""
VibeVoice engine handler with special RAM management
"""

import torch
import gc
from typing import Optional, TYPE_CHECKING

from .generic_handler import GenericHandler

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
                        import gc
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
        VibeVoice full unload with complete deletion.
        
        For VibeVoice, performs complete deletion instead of CPU migration
        to prevent system RAM accumulation.
        """
        if memory_to_free is not None and memory_to_free < wrapper.loaded_size():
            # Try partial unload first
            freed = self.partially_unload(wrapper, 'cpu', memory_to_free)
            success = freed >= memory_to_free
            print(f"{'✅' if success else '❌'} Partial unload: freed {freed // 1024 // 1024}MB (requested {memory_to_free // 1024 // 1024}MB)")
            return success
            
        # Full unload - for VibeVoice, delete completely instead of moving to CPU
        freed = 0
        model_location = "unknown"
        
        if hasattr(wrapper, 'model') and wrapper.model is not None:
            try:
                # Detect if model is on GPU or CPU
                if hasattr(wrapper.model, 'device'):
                    model_location = str(wrapper.model.device)
                elif wrapper._is_loaded_on_gpu:
                    model_location = "GPU"
                else:
                    model_location = "CPU"
                
                print(f"🔍 VibeVoice deletion: Model currently on {model_location}")
                
                # Clear VibeVoice class-level cache first
                if hasattr(wrapper.model, '__class__'):
                    model_class = wrapper.model.__class__
                    if hasattr(model_class, '_shared_model'):
                        model_class._shared_model = None
                        model_class._shared_processor = None
                        model_class._shared_config = None
                        model_class._shared_model_name = None
                        print(f"🗑️ Cleared VibeVoice class-level cache")
                
                # Estimate memory before deletion (same size regardless of location)
                freed = wrapper._memory_size
                
                # Delete the model completely (works for both GPU and CPU)
                del wrapper.model
                wrapper.model = None
                
                # Force garbage collection to actually free memory
                try:
                    import gc
                    gc.collect()
                except Exception as gc_error:
                    print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")
                
                # Clear CUDA cache if model was on GPU - AGGRESSIVE CLEANUP
                if model_location.lower() in ['gpu', 'cuda'] or 'cuda' in model_location.lower():
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        # Force multiple cleanup passes
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all operations to complete
                        torch.cuda.empty_cache()  # Second pass after sync
                        
                        # Force garbage collection multiple times
                        for _ in range(3):  # Multiple GC passes
                            try:
                                import gc
                                gc.collect()
                            except Exception as gc_error:
                                print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")
                            torch.cuda.empty_cache()
                        
                        # NUCLEAR OPTION: Force CUDA device reset
                        try:
                            print(f"⚠️ NUCLEAR: Forcing CUDA device reset to clear stubborn VibeVoice memory")
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.empty_cache()
                            
                            # Try ComfyUI model management
                            try:
                                from comfy import model_management
                                if hasattr(model_management, 'free_memory'):
                                    model_management.free_memory(8 * 1024 * 1024 * 1024, torch.cuda.current_device())  # Request 8GB
                                    print(f"🧹 ComfyUI freed memory")
                            except:
                                pass
                            
                            # Final synchronization
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"⚠️ CUDA reset failed: {e}")
                        
                        print(f"🧹 NUCLEAR CUDA cleanup completed")
                
                print(f"🗑️ VibeVoice: Model deleted completely from {model_location}")
                
            except Exception as e:
                print(f"⚠️ VibeVoice deletion error: {e}")
        
        success = freed > 0
        print(f"{'✅' if success else '❌'} VibeVoice full deletion: freed {freed // 1024 // 1024}MB from {model_location}")
        return success