"""
Generic engine handler for most TTS engines (ChatterBox, F5-TTS, RVC, etc.)
"""

import torch
import gc
from typing import Optional, TYPE_CHECKING

from .base_handler import BaseEngineHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class GenericHandler(BaseEngineHandler):
    """
    Generic handler for most TTS engines (ChatterBox, F5-TTS, RVC, etc.).
    
    Handles standard CPU migration and device movement for engines using
    the default unloading behavior.
    """
    
    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """Standard CPU migration for generic engines"""
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return 0
            
        freed_memory = 0
        
        try:
            # Move model to CPU if it has a .to() method
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                    freed_memory = wrapper._memory_size
                    wrapper.current_device = 'cpu'
                    wrapper._is_loaded_on_gpu = False
                    print(f"🔄 Moved {wrapper.model_info.model_type} model ({wrapper.model_info.engine}) to CPU, freed {freed_memory // 1024 // 1024}MB")
                except Exception as e:
                    print(f"⚠️ Failed to move {wrapper.model_info.model_type} model to CPU: {e}")
                    # Still mark as unloaded if the model reported an error moving to CPU
                    wrapper.current_device = 'cpu'
                    wrapper._is_loaded_on_gpu = False
                    freed_memory = wrapper._memory_size
                
            # Handle nested models (like ChatterBox with multiple components)
            elif hasattr(model, '__dict__'):
                for attr_name, attr_value in model.__dict__.items():
                    if hasattr(attr_value, 'to') and hasattr(attr_value, 'parameters'):
                        try:
                            attr_value.to('cpu')
                            freed_memory += self._estimate_model_memory(attr_value)
                        except Exception as e:
                            print(f"⚠️ Failed to move {attr_name} to CPU: {e}")
                            pass
                            
                if freed_memory > 0:
                    wrapper.current_device = 'cpu' 
                    wrapper._is_loaded_on_gpu = False
                    print(f"🔄 Moved {wrapper.model_info.model_type} model components ({wrapper.model_info.engine}) to CPU, freed {freed_memory // 1024 // 1024}MB")
                    
        except Exception as e:
            print(f"⚠️ Failed to partially unload {wrapper.model_info.model_type} model: {e}")
            
        # Force garbage collection after unloading
        if freed_memory > 0:
            try:
                import gc
                gc.collect()
            except Exception as gc_error:
                print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ CUDA cache clear warning (safe to ignore): {e}")
                
        return freed_memory
    
    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """Standard full unload using CPU migration"""
        if memory_to_free is not None and memory_to_free < wrapper.loaded_size():
            # Try partial unload first
            freed = self.partially_unload(wrapper, 'cpu', memory_to_free)
            success = freed >= memory_to_free
            print(f"{'✅' if success else '❌'} Partial unload: freed {freed // 1024 // 1024}MB (requested {memory_to_free // 1024 // 1024}MB)")
            return success
            
        # Full unload - use standard CPU migration
        freed = self.partially_unload(wrapper, 'cpu', wrapper._memory_size)
        success = freed > 0
        print(f"{'✅' if success else '❌'} Full unload: freed {freed // 1024 // 1024}MB")
        return success
    
    @staticmethod
    def _estimate_model_memory(model) -> int:
        """Estimate memory usage of a PyTorch model"""
        if not hasattr(model, 'parameters'):
            return 0
            
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size