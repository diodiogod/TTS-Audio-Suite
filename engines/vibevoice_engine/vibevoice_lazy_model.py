"""
Lazy VibeVoice Model - Only creates layers when accessed
This is a radical approach to avoid the full model construction bottleneck
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging


class LazyVibeVoiceModel(nn.Module):
    """
    A lazy VibeVoice model that only creates layers as they're accessed
    This completely bypasses the standard constructor
    """
    
    def __init__(self, gguf_state_dict: Dict[str, Any], config: dict, device='cpu'):
        super().__init__()
        self.gguf_state_dict = gguf_state_dict
        self.config = config
        self.device = device
        self._modules = {}
        self._loaded_modules = set()
        
        print(f"⚡ Created lazy VibeVoice model (no actual layers yet)")
        
        # Store the state dict keys organized by module
        self.module_keys = self._organize_state_dict_keys(gguf_state_dict)
        
        # Create module stubs
        self._create_module_stubs()
    
    def _organize_state_dict_keys(self, state_dict):
        """Organize state dict keys by module"""
        modules = {}
        for key in state_dict.keys():
            # Extract module name (e.g., "model.language_model.layers.0")
            parts = key.split('.')
            if len(parts) > 1:
                module_path = '.'.join(parts[:-1])  # Everything except the parameter name
                if module_path not in modules:
                    modules[module_path] = []
                modules[module_path].append(key)
        
        print(f"📊 Organized {len(state_dict)} tensors into {len(modules)} modules")
        return modules
    
    def _create_module_stubs(self):
        """Create placeholder modules that will be loaded on demand"""
        # Create top-level placeholders
        self.model = LazySubModule(self, 'model', self.module_keys, self.gguf_state_dict)
        self.lm_head = LazySubModule(self, 'lm_head', self.module_keys, self.gguf_state_dict)
        
        print(f"✅ Created lazy module structure")
    
    def forward(self, *args, **kwargs):
        """Forward pass - loads modules as needed"""
        print(f"⚠️ LazyVibeVoiceModel forward called - loading required modules...")
        
        # This would need to implement the actual forward logic
        # For now, we'll raise an error to show the model was created
        raise NotImplementedError("LazyVibeVoiceModel forward not yet implemented")
    
    def generate(self, *args, **kwargs):
        """Generate method - loads modules as needed"""
        print(f"⚠️ LazyVibeVoiceModel generate called - loading required modules...")
        
        # This would need to implement the actual generate logic
        raise NotImplementedError("LazyVibeVoiceModel generate not yet implemented")
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to handle GGUF tensors"""
        # Store the state dict without actually loading into modules
        self.gguf_state_dict.update(state_dict)
        print(f"📦 Updated lazy model with {len(state_dict)} tensors")
        
        # Return empty lists for missing/unexpected keys
        return [], []
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        # Don't actually move anything yet - will move on demand
        print(f"📤 Set target device to {device} (lazy movement)")
        return self
    
    def eval(self):
        """Set to eval mode"""
        self.training = False
        return self


class LazySubModule(nn.Module):
    """A lazy submodule that creates itself on first access"""
    
    def __init__(self, parent, name, module_keys, gguf_state_dict):
        super().__init__()
        self.parent = parent
        self.name = name
        self.module_keys = module_keys
        self.gguf_state_dict = gguf_state_dict
        self._materialized = False
        self._real_module = None
    
    def __getattr__(self, item):
        if item.startswith('_'):
            return super().__getattr__(item)
        
        if not self._materialized:
            print(f"🔄 Lazy loading module: {self.name}.{item}")
            # This is where we would materialize the actual module
            # For now, return another lazy module
            return LazySubModule(self, f"{self.name}.{item}", self.module_keys, self.gguf_state_dict)
        
        if self._real_module:
            return getattr(self._real_module, item)
        
        return super().__getattr__(item)
    
    def forward(self, *args, **kwargs):
        """Forward pass - materialize if needed"""
        if not self._materialized:
            print(f"⚠️ Materializing {self.name} for forward pass...")
            # This would materialize the actual module
        
        if self._real_module:
            return self._real_module(*args, **kwargs)
        
        # Placeholder
        return args[0] if args else None


def create_lazy_vibevoice_from_gguf(model_path: str, device: torch.device) -> tuple:
    """
    Create a lazy VibeVoice model that avoids full construction
    """
    from pathlib import Path
    from .gguf_loader import VibeVoiceGGUFLoader
    import json
    
    print(f"💨 Creating lazy VibeVoice model (instant loading)...")
    
    # Load GGUF data
    gguf_path = Path(model_path) / "model.gguf"
    config_path = Path(model_path) / "config.json"
    
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
    
    # Load GGUF tensors (ignore config.json - use tensor-based detection instead)
    loader = VibeVoiceGGUFLoader()
    state_dict, config = loader.load_gguf_model(
        str(gguf_path),
        None,  # No config path
        keep_quantized=True,
        skip_config=True  # Skip config.json loading, use tensor-based detection
    )
    
    if not config:
        raise RuntimeError("Failed to detect VibeVoice config from GGUF tensors")
    
    print(f"📦 Loaded {len(state_dict)} GGUF tensors")
    
    # Create lazy model - this should be instant
    lazy_model = LazyVibeVoiceModel(state_dict, config, device)
    
    # Move to device (lazy)
    lazy_model = lazy_model.to(device)
    lazy_model.eval()
    
    print(f"✅ Lazy VibeVoice model ready (instant!)")
    
    # Create a wrapper that looks like a real model
    return create_model_wrapper(lazy_model), True


def create_model_wrapper(lazy_model):
    """Create a wrapper that makes the lazy model look like a real VibeVoice model"""
    
    class VibeVoiceModelWrapper:
        def __init__(self, lazy_model):
            self.lazy_model = lazy_model
            self.model = lazy_model.model  # Lazy submodule
            self.lm_head = lazy_model.lm_head  # Lazy submodule
            self.device = lazy_model.device
            self._materialized_model = None  # Cache the materialized model
            
        def to(self, device):
            self.lazy_model = self.lazy_model.to(device)
            self.device = device
            return self
        
        def cuda(self, device=None):
            """CUDA method for compatibility"""
            if device is None:
                device = 'cuda'
            return self.to(device)
        
        def cpu(self):
            """CPU method for compatibility"""
            return self.to('cpu')
        
        def eval(self):
            self.lazy_model.eval()
            return self
        
        def parameters(self):
            """Return parameters - create a dummy parameter for device detection"""
            # Create a dummy parameter on the target device so next(model.parameters()).device works
            dummy_param = torch.nn.Parameter(torch.tensor([0.0], device=self.device))
            yield dummy_param
        
        def generate(self, *args, **kwargs):
            """
            Generate method - materialize once, cache forever
            """
            if self._materialized_model is None:
                print(f"🔄 First generation: materializing GGUF model (one-time 90s cost)...")
                self._materialized_model = self._create_gguf_model_cached()
            else:
                print(f"⚡ Using cached materialized model (instant!)")
            
            # Apply stored inference steps
            if hasattr(self, '_inference_steps'):
                if hasattr(self._materialized_model, 'set_ddpm_inference_steps'):
                    self._materialized_model.set_ddpm_inference_steps(num_steps=self._inference_steps)
                elif hasattr(self._materialized_model, 'model') and hasattr(self._materialized_model.model, 'set_ddpm_inference_steps'):
                    self._materialized_model.model.set_ddpm_inference_steps(num_steps=self._inference_steps)
                print(f"🔄 Applied stored inference steps: {self._inference_steps}")
            
            # Delegate to the cached model
            return self._materialized_model.generate(*args, **kwargs)
        
        def _create_gguf_model_cached(self):
            """Use ComfyUI-style model detection - no manual construction!"""
            from .vibevoice_model_detection import load_vibevoice_from_gguf_state_dict
            from .gguf_vibevoice_ops import GGMLOps
            import time
            
            print(f"🚀 Using ComfyUI-style GGUF model detection (fast approach)...")
            start_time = time.time()
            
            # Create GGUF ops
            ops = GGMLOps()
            
            # Use ComfyUI's approach - model detection + custom ops
            model = load_vibevoice_from_gguf_state_dict(
                self.lazy_model.gguf_state_dict,
                model_options={"custom_operations": ops}
            )
            
            if model is None:
                raise RuntimeError("Failed to detect VibeVoice model from GGUF using ComfyUI approach")
            
            # Keep on CPU to save VRAM (GGUF tensors move to GPU on-demand)
            print(f"⚡ Model stays on CPU (GGUF tensors move to GPU during computation)")
            model.eval()
            
            print(f"✅ VibeVoice GGUF model ready in {time.time() - start_time:.1f}s using ComfyUI approach!")
            print(f"🎯 Future generations will be instant (cached model)")
            
            return model
        
        def __call__(self, *args, **kwargs):
            return self.lazy_model(*args, **kwargs)
        
        def set_ddpm_inference_steps(self, num_steps):
            """Set inference steps - will be applied when model is materialized"""
            print(f"🔄 Storing inference steps: {num_steps} (will apply on materialization)")
            self._inference_steps = num_steps
        
        def __getattr__(self, name):
            """Delegate unknown attributes to the lazy model"""
            return getattr(self.lazy_model, name)
    
    return VibeVoiceModelWrapper(lazy_model)