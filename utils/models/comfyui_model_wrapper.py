"""
ComfyUI Model Wrapper for TTS Audio Suite

This module provides wrappers that make TTS models compatible with ComfyUI's
model management system, enabling automatic memory management, "Clear VRAM" 
button functionality, and proper integration with ComfyUI's model lifecycle.
"""

import torch
import weakref
import gc
from typing import Optional, Any, Dict, Union
from dataclasses import dataclass

# Import ComfyUI's model management if available
try:
    import comfy.model_management as model_management
    COMFYUI_AVAILABLE = True
except ImportError:
    # Fallback if ComfyUI not available
    COMFYUI_AVAILABLE = False
    model_management = None


@dataclass 
class ModelInfo:
    """Information about a model for memory management"""
    model: Any
    model_type: str  # "tts", "vc", "audio_separation", "hubert", etc.
    engine: str      # "chatterbox", "f5tts", "higgs_audio", "rvc", etc.
    device: str
    memory_size: int  # in bytes
    load_device: str


class ComfyUIModelWrapper:
    """
    Wrapper that makes TTS models compatible with ComfyUI's model management system.
    
    This allows TTS models to be automatically unloaded when VRAM is low,
    work with "Clear VRAM" buttons, and integrate properly with ComfyUI's ecosystem.
    """
    
    def __init__(self, model: Any, model_info: ModelInfo):
        """
        Initialize the wrapper.
        
        Args:
            model: The actual model instance (ChatterBox, F5-TTS, etc.)
            model_info: Metadata about the model
        """
        self.model = model
        self.model_info = model_info
        self.load_device = model_info.load_device
        self.current_device = model_info.device
        self._memory_size = model_info.memory_size
        
        # ComfyUI compatibility attributes
        self.device = model_info.device
        self.dtype = getattr(model, 'dtype', torch.float32)
        
        # ComfyUI expects these attributes for diffusion models (TTS models don't need them)
        self.model_patches_models = []  # Empty list for TTS models
        self.parent = None              # TTS models don't have parent models
        
        # Track if model is currently loaded on GPU
        self._is_loaded_on_gpu = self.current_device not in ['cpu', 'offload']
        
        # Keep weak reference to avoid circular references
        self._model_ref = weakref.ref(model) if model is not None else None
        
    def loaded_size(self) -> int:
        """Return the memory size of the model in bytes"""
        if self._is_loaded_on_gpu:
            return self._memory_size
        return 0
        
    def model_size(self) -> int:
        """Return the total model size in bytes"""
        return self._memory_size
    
    def partially_unload(self, device: str, memory_to_free: int) -> int:
        """
        Partially unload the model to free memory.
        
        For TTS models, this typically means moving to CPU or offloading.
        
        Args:
            device: Target device to move to (usually 'cpu')
            memory_to_free: Amount of memory to free in bytes
            
        Returns:
            Amount of memory actually freed in bytes
        """
        if not self._is_loaded_on_gpu:
            return 0
            
        model = self._model_ref() if self._model_ref else None
        if model is None:
            return 0
            
        freed_memory = 0
        
        try:
            # Move model to CPU if it has a .to() method
            if hasattr(model, 'to'):
                model.to('cpu')
                freed_memory = self._memory_size
                self.current_device = 'cpu'
                self._is_loaded_on_gpu = False
                print(f"🔄 Moved {self.model_info.model_type} model ({self.model_info.engine}) to CPU, freed {freed_memory // 1024 // 1024}MB")
                
            # Handle nested models (like ChatterBox with multiple components)
            elif hasattr(model, '__dict__'):
                for attr_name, attr_value in model.__dict__.items():
                    if hasattr(attr_value, 'to') and hasattr(attr_value, 'parameters'):
                        try:
                            attr_value.to('cpu')
                            freed_memory += self._estimate_model_memory(attr_value)
                        except Exception:
                            pass
                            
                if freed_memory > 0:
                    self.current_device = 'cpu' 
                    self._is_loaded_on_gpu = False
                    print(f"🔄 Moved {self.model_info.model_type} model components ({self.model_info.engine}) to CPU, freed {freed_memory // 1024 // 1024}MB")
                    
        except Exception as e:
            print(f"⚠️ Failed to partially unload {self.model_info.model_type} model: {e}")
            
        # Force garbage collection after unloading
        if freed_memory > 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return freed_memory
    
    def model_unload(self, memory_to_free: Optional[int] = None, unpatch_weights: bool = True) -> bool:
        """
        Fully unload the model from GPU memory.
        
        Args:
            memory_to_free: Amount of memory to free (ignored for full unload)
            unpatch_weights: Whether to unpatch weights (TTS models don't use this)
            
        Returns:
            True if model was unloaded, False otherwise
        """
        if memory_to_free is not None and memory_to_free < self.loaded_size():
            # Try partial unload first
            freed = self.partially_unload('cpu', memory_to_free)
            return freed >= memory_to_free
            
        # Full unload
        return self.partially_unload('cpu', self._memory_size) > 0
    
    def model_load(self, device: Optional[str] = None) -> None:
        """
        Load the model back to GPU.
        
        Args:
            device: Device to load to (defaults to original load_device)
        """
        if self._is_loaded_on_gpu:
            return
            
        target_device = device or self.load_device
        model = self._model_ref() if self._model_ref else None
        
        if model is None:
            return
            
        try:
            # Move model back to GPU
            if hasattr(model, 'to'):
                model.to(target_device)
                self.current_device = target_device
                self._is_loaded_on_gpu = True
                print(f"🔄 Moved {self.model_info.model_type} model ({self.model_info.engine}) back to {target_device}")
                
            # Handle nested models  
            elif hasattr(model, '__dict__'):
                for attr_name, attr_value in model.__dict__.items():
                    if hasattr(attr_value, 'to') and hasattr(attr_value, 'parameters'):
                        try:
                            attr_value.to(target_device)
                        except Exception:
                            pass
                            
                self.current_device = target_device
                self._is_loaded_on_gpu = True
                print(f"🔄 Moved {self.model_info.model_type} model components ({self.model_info.engine}) back to {target_device}")
                
        except Exception as e:
            print(f"⚠️ Failed to load {self.model_info.model_type} model to {target_device}: {e}")
    
    def is_clone(self, other) -> bool:
        """Check if this model is a clone of another model"""
        if not isinstance(other, ComfyUIModelWrapper):
            return False
        return (self.model_info.model_type == other.model_info.model_type and 
                self.model_info.engine == other.model_info.engine)
    
    def detach(self, unpatch_all: bool = False) -> None:
        """Detach the model (TTS models don't need special detaching)"""
        pass
        
    @staticmethod
    def _estimate_model_memory(model) -> int:
        """Estimate memory usage of a PyTorch model"""
        if not hasattr(model, 'parameters'):
            return 0
            
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size
    
    @staticmethod 
    def calculate_model_memory(model: Any) -> int:
        """Calculate total memory usage of a model in bytes"""
        if hasattr(model, 'parameters'):
            # PyTorch model
            return ComfyUIModelWrapper._estimate_model_memory(model)
        elif hasattr(model, '__dict__'):
            # Complex model with multiple components
            total_size = 0
            for attr_value in model.__dict__.values():
                if hasattr(attr_value, 'parameters'):
                    total_size += ComfyUIModelWrapper._estimate_model_memory(attr_value)
            return total_size
        else:
            # Estimate based on common model sizes
            return 1024 * 1024 * 1024  # Default 1GB estimate
    
    def __repr__(self):
        return f"ComfyUIModelWrapper({self.model_info.model_type}:{self.model_info.engine}, {self._memory_size // 1024 // 1024}MB, device={self.current_device})"


class ComfyUITTSModelManager:
    """
    Manager that integrates TTS models with ComfyUI's model management system.
    
    This replaces static caches with ComfyUI-managed model loading/unloading.
    """
    
    def __init__(self):
        self._model_cache: Dict[str, ComfyUIModelWrapper] = {}
        
    def load_model(self, 
                   model_factory_func, 
                   model_key: str,
                   model_type: str,
                   engine: str, 
                   device: str,
                   **factory_kwargs) -> ComfyUIModelWrapper:
        """
        Load a model using ComfyUI's model management system.
        
        Args:
            model_factory_func: Function that creates the model
            model_key: Unique key for caching
            model_type: Type of model ("tts", "vc", etc.)  
            engine: Engine name ("chatterbox", "f5tts", etc.)
            device: Target device
            **factory_kwargs: Arguments for model factory function
            
        Returns:
            ComfyUI-wrapped model
        """
        # Check if already cached
        if model_key in self._model_cache:
            wrapper = self._model_cache[model_key]
            # Ensure model is loaded on correct device
            if wrapper.current_device != device and device != 'auto':
                wrapper.model_load(device)
            return wrapper
            
        # Create the model
        print(f"🔧 Creating new {model_type} model ({engine}) on {device}")
        # Ensure device parameter is available to factory function
        factory_kwargs['device'] = device
        model = model_factory_func(**factory_kwargs)
        
        # Calculate memory usage
        memory_size = ComfyUIModelWrapper.calculate_model_memory(model)
        
        # Create model info
        model_info = ModelInfo(
            model=model,
            model_type=model_type,
            engine=engine,
            device=device,
            memory_size=memory_size,
            load_device=device
        )
        
        # Wrap for ComfyUI
        wrapper = ComfyUIModelWrapper(model, model_info)
        
        # Cache the wrapper
        self._model_cache[model_key] = wrapper
        
        # Register with ComfyUI if available
        if COMFYUI_AVAILABLE and model_management is not None:
            try:
                # Use ComfyUI's model loading system
                model_management.load_models_gpu([wrapper])
                print(f"✅ Registered {model_type} model with ComfyUI model management")
            except Exception as e:
                print(f"⚠️ Failed to register with ComfyUI model management: {e}")
                
        return wrapper
    
    def get_model(self, model_key: str) -> Optional[ComfyUIModelWrapper]:
        """Get a cached model by key"""
        return self._model_cache.get(model_key)
        
    def remove_model(self, model_key: str) -> bool:
        """Remove a model from cache"""
        if model_key in self._model_cache:
            wrapper = self._model_cache.pop(model_key)
            # Unload from GPU
            wrapper.model_unload()
            return True
        return False
        
    def clear_cache(self, model_type: Optional[str] = None, engine: Optional[str] = None):
        """Clear cached models with optional filtering"""
        keys_to_remove = []
        
        for key, wrapper in self._model_cache.items():
            should_remove = True
            
            if model_type and wrapper.model_info.model_type != model_type:
                should_remove = False
            if engine and wrapper.model_info.engine != engine:  
                should_remove = False
                
            if should_remove:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            self.remove_model(key)
            
        print(f"🧹 Cleared {len(keys_to_remove)} models from cache")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_memory = sum(w.loaded_size() for w in self._model_cache.values())
        by_type = {}
        by_engine = {}
        
        for wrapper in self._model_cache.values():
            model_type = wrapper.model_info.model_type
            engine = wrapper.model_info.engine
            
            by_type[model_type] = by_type.get(model_type, 0) + 1
            by_engine[engine] = by_engine.get(engine, 0) + 1
            
        return {
            'total_models': len(self._model_cache),
            'total_memory_mb': total_memory // 1024 // 1024,
            'by_type': by_type,
            'by_engine': by_engine,
            'comfyui_integration': COMFYUI_AVAILABLE
        }


# Global instance for all TTS models
tts_model_manager = ComfyUITTSModelManager()