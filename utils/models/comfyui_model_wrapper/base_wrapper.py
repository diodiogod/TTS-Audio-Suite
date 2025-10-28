"""
Base ComfyUI model wrapper class
"""

import torch
import weakref
import gc
from typing import Optional, Any
from dataclasses import dataclass

from .cache_utils import invalidate_all_caches

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


class SimpleModelWrapper:
    """
    Simple wrapper to add .model attribute for ComfyUI compatibility.
    ComfyUI expects .model.model for logging, so we wrap simple PyTorch models.
    """
    def __init__(self, model):
        self.model = model  # This is what ComfyUI accesses for logging

    def to(self, device):
        """Forward .to() calls to the wrapped model"""
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        return self

    def __getattr__(self, name):
        """Forward all other attributes to the wrapped model"""
        return getattr(self.model, name)


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
        # Convert device to torch.device object for ComfyUI compatibility
        device_name = model_info.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        
        # ComfyUI expects torch.device objects, not strings
        if isinstance(device_name, str):
            if device_name == "cuda":
                self.device = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_available() else 0)
            else:
                self.device = torch.device(device_name)
        else:
            self.device = device_name
        self.dtype = getattr(model, 'dtype', torch.float32)
        self.offload_device = 'cpu'  # TTS models offload to CPU
        
        # ComfyUI compatibility attributes for diffusion models (TTS models don't need them)
        self.model_patches_models = []  # Empty list for TTS models
        self.parent = None              # TTS models don't have parent models
        
        # Additional ComfyUI LoadedModel compatibility attributes  
        # Use the same torch.device object for load_device
        self.load_device = self.device
        self.currently_used = True
        self.model_finalizer = None  # Will be set by LoadedModel
        self._patcher_finalizer = None
        
        # ComfyUI model patcher attributes (required for load_models_gpu)
        self.model_patches_to = {}  # Patch mapping for diffusion models (empty for TTS)
        self.model_options = {}     # Model loading options
        self.model_keys = set()     # Model state dict keys
        
        # Track if model is currently loaded on GPU
        self._is_loaded_on_gpu = self.current_device not in ['cpu', 'offload']
        
        # Track if model is valid for reuse (false if corrupted by CPU offloading)
        self._is_valid_for_reuse = True
        
        # Keep weak reference to avoid circular references
        self._model_ref = weakref.ref(model) if model is not None else None
        
    def loaded_size(self) -> int:
        """Return the memory size of the model in bytes"""
        size = self._memory_size if self._is_loaded_on_gpu else 0
        return size
        
    def model_size(self) -> int:
        """Return the total model size in bytes"""
        return self._memory_size

    def model_memory(self) -> int:
        """Return the total model size in bytes (alias for model_size for ComfyUI compatibility)"""
        return self._memory_size

    def model_offloaded_memory(self) -> int:
        """Return the amount of memory that would be freed if offloaded"""
        return self.model_size() - self.loaded_size()
    
    def current_loaded_device(self) -> str:
        """Return the current device the model is loaded on"""
        return self.current_device
    
    def partially_unload(self, device: str, memory_to_free: int) -> int:
        """
        Partially unload the model to free memory.

        Uses engine-specific handlers for specialized behavior.

        Args:
            device: Target device to move to (usually 'cpu')
            memory_to_free: Amount of memory to free in bytes

        Returns:
            Amount of memory actually freed in bytes
        """
        if not self._is_loaded_on_gpu:
            print(f"⚠️ Skipping unload: model already marked as not on GPU (_is_loaded_on_gpu={self._is_loaded_on_gpu})")
            return 0

        # Import and use engine-specific handler
        from .engine_handlers import get_engine_handler
        handler = get_engine_handler(self.model_info.engine)

        return handler.partially_unload(self, device, memory_to_free)
    
    def model_unload(self, memory_to_free: Optional[int] = None, unpatch_weights: bool = True) -> bool:
        """
        Fully unload the model from GPU memory.

        Args:
            memory_to_free: Amount of memory to free (ignored for full unload)
            unpatch_weights: Whether to unpatch weights (TTS models don't use this)

        Returns:
            True if model was unloaded, False otherwise
        """
        print(f"🔄 TTS Model unload requested: {self.model_info.engine} {self.model_info.model_type} (_is_loaded_on_gpu={self._is_loaded_on_gpu})")

        # Import and use engine-specific handler
        from .engine_handlers import get_engine_handler
        handler = get_engine_handler(self.model_info.engine)

        return handler.model_unload(self, memory_to_free, unpatch_weights)
    
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
            # Move model back to GPU (comprehensive approach)
            if hasattr(model, 'to'):
                model.to(target_device)
                print(f"🔄 Moved main {self.model_info.model_type} model ({self.model_info.engine}) to {target_device}")

            # CRITICAL: Recursively move ALL nested components to ensure device consistency
            self._move_all_components_to_device(model, target_device, depth=0)

            self.current_device = target_device
            self._is_loaded_on_gpu = True
            print(f"✅ Fully moved {self.model_info.model_type} model components ({self.model_info.engine}) back to {target_device}")

            # Re-register with ComfyUI after reload so "Clear VRAM" continues to work
            try:
                import comfy.model_management as model_management
                # Check if we're in current_loaded_models list
                if hasattr(model_management, 'current_loaded_models'):
                    # Find our LoadedModel wrapper
                    found = False
                    for loaded_model in model_management.current_loaded_models:
                        if hasattr(loaded_model, 'model') and loaded_model.model == self:
                            found = True
                            break

                    # If not found, re-register
                    if not found:
                        loaded_model = model_management.LoadedModel(self)
                        loaded_model.real_model = weakref.ref(model)
                        loaded_model._tts_wrapper_ref = self
                        # Set up finalizer that ComfyUI expects
                        if hasattr(model_management, 'cleanup_models'):
                            loaded_model.model_finalizer = weakref.finalize(model, model_management.cleanup_models)
                        else:
                            loaded_model.model_finalizer = weakref.finalize(model, lambda: None)
                        model_management.current_loaded_models.insert(0, loaded_model)
                        print(f"✅ Re-registered with ComfyUI model management after reload")
            except Exception as e:
                print(f"⚠️ Could not re-register with ComfyUI ({e}) - 'Clear VRAM' may not work")

        except Exception as e:
            print(f"⚠️ Error moving model to {target_device}: {e}")
    
    def _move_all_components_to_device(self, obj, target_device: str, depth: int = 0, max_depth: int = 5):
        """
        Recursively move all PyTorch components to target device.
        This ensures no CPU/GPU device mismatches after model reload.
        """
        if depth > max_depth:
            return

        if obj is None:
            return

        # Move PyTorch modules (any object with .to() method)
        if hasattr(obj, 'to') and callable(getattr(obj, 'to')):
            try:
                obj.to(target_device)
                if depth <= 1:  # Log top-level and first-level components
                    print(f"  {'  ' * depth}📦 Moved {type(obj).__name__} to {target_device}")
            except Exception as e:
                if depth <= 1:
                    print(f"  {'  ' * depth}⚠️ Failed to move {type(obj).__name__}: {e}")

        # Recurse through object attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                # Skip private attributes and known problematic ones
                if attr_name.startswith('_'):
                    continue
                if attr_value is None:
                    continue
                # Skip internal PyTorch structures (already handled by .to())
                if attr_name in ['_modules', '_parameters', '_buffers', 'training']:
                    continue
                try:
                    self._move_all_components_to_device(attr_value, target_device, depth + 1, max_depth)
                except Exception:
                    pass
    
    def is_clone(self, other) -> bool:
        """Check if this model is a clone of another model"""
        if not isinstance(other, ComfyUIModelWrapper):
            return False
        return (self.model_info.model_type == other.model_info.model_type and
                self.model_info.engine == other.model_info.engine)

    def is_dead(self) -> bool:
        """Check if the model's weak reference is dead (for ComfyUI cleanup)"""
        if self._model_ref is None:
            return True
        return self._model_ref() is None
    
    def detach(self, unpatch_all: bool = False) -> None:
        """Detach the model - actually unload from GPU to CPU and invalidate cache"""
        print(f"🔧 TTS Model detach called: {self.model_info.engine} {self.model_info.model_type} (unpatch_all={unpatch_all})")
        
        # Actually unload the model from GPU
        freed = self.partially_unload('cpu', self._memory_size)
        if freed > 0:
            print(f"✅ TTS Model detached: freed {freed // 1024 // 1024}MB VRAM")
        else:
            print(f"⚠️ TTS Model detach: no memory freed (model may already be on CPU)")
        
        # CRITICAL: Mark model as invalid to prevent reuse of corrupted state
        # Only needed for engines with CUDA graphs (Higgs Audio) that cannot be safely reused after CPU offloading
        if self.model_info.engine == "higgs_audio":
            self._is_valid_for_reuse = False
            print(f"🚫 Marked {self.model_info.engine} model as invalid for reuse (CUDA graphs corrupted by CPU migration)")

            # CRITICAL: Clear node-level engine caches to prevent reuse of corrupted engines
            # This is essential because TTS nodes have their own caching separate from ComfyUI wrapper cache
            invalidate_all_caches()
        else:
            # Other engines (ChatterBox, F5-TTS, VibeVoice) can be safely reused after device movement
            print(f"✅ {self.model_info.engine} model detached, marked valid for reuse")
    
    def partially_load(self, device, extra_memory, force_patch_weights=False):
        """
        Partially load model to device (ComfyUI compatibility method)
        
        Args:
            device: Target device
            extra_memory: Extra memory available
            force_patch_weights: Whether to force patch weights
            
        Returns:
            Amount of memory used
        """
        print(f"🔄 TTS Model partially_load requested: {self.model_info.engine} {self.model_info.model_type} to {device}")
        
        # For TTS models, we either fully load or fully unload
        if device != 'cpu' and not self._is_loaded_on_gpu:
            self.model_load(device)
            return self._memory_size
        
        return 0  # No additional memory used
        
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
            for attr_name, attr_value in model.__dict__.items():
                if hasattr(attr_value, 'parameters'):
                    total_size += ComfyUIModelWrapper._estimate_model_memory(attr_value)
                elif attr_name == 'f5tts_model' and hasattr(attr_value, '__dict__'):
                    # Special handling for ChatterBoxF5TTS wrapper
                    for sub_attr_name, sub_attr_value in attr_value.__dict__.items():
                        if hasattr(sub_attr_value, 'parameters'):
                            total_size += ComfyUIModelWrapper._estimate_model_memory(sub_attr_value)
            return total_size if total_size > 0 else 1024 * 1024 * 1024  # 1GB default if nothing found
        else:
            # Estimate based on common model sizes
            return 1024 * 1024 * 1024  # Default 1GB estimate
    
    def __repr__(self):
        return f"ComfyUIModelWrapper({self.model_info.model_type}:{self.model_info.engine}, {self._memory_size // 1024 // 1024}MB, device={self.current_device})"