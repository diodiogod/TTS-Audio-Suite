"""
Direct GGUF Loader - Load tensors directly without model construction
Creates a minimal model wrapper that just holds the tensors
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

class MinimalVibeVoiceWrapper(nn.Module):
    """
    Minimal wrapper that holds GGUF tensors without full model construction
    """
    def __init__(self, state_dict, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Store tensors as parameters without creating full architecture
        self.tensor_storage = nn.ParameterDict()
        
        print(f"📦 Storing {len(state_dict)} GGUF tensors in minimal wrapper...")
        
        # Convert GGUF tensors to parameters with dimension adaptation
        from .gguf_utils import dequantize_tensor
        
        for key, gguf_tensor in state_dict.items():
            # Clean key name for parameter dict
            clean_key = key.replace('.', '_').replace('-', '_')
            
            # Store the tensor (keep quantized if possible)
            if hasattr(gguf_tensor, 'tensor_type'):
                # Keep quantized
                self.tensor_storage[clean_key] = nn.Parameter(gguf_tensor, requires_grad=False)
            else:
                # Regular tensor
                self.tensor_storage[clean_key] = nn.Parameter(gguf_tensor, requires_grad=False)
        
        print(f"✅ Minimal wrapper created with {len(self.tensor_storage)} tensors")
    
    def forward(self, *args, **kwargs):
        """
        Delegate to actual VibeVoice forward when needed
        This will be implemented when we need inference
        """
        raise NotImplementedError("Minimal wrapper doesn't support forward pass yet")
    
    def to(self, *args, **kwargs):
        """Override to handle device movement efficiently"""
        result = super().to(*args, **kwargs)
        if len(args) > 0 and hasattr(args[0], 'type'):
            result.device = args[0]
        return result


def load_gguf_direct(
    model_path: str,
    device: torch.device
) -> Tuple[Any, bool]:
    """
    Load GGUF directly without full model construction
    """
    from .gguf_loader import VibeVoiceGGUFLoader
    
    print(f"⚡ Direct GGUF loading - no model construction...")
    
    # Load GGUF data
    gguf_path = Path(model_path) / "model.gguf"
    config_path = Path(model_path) / "config.json"
    
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
    
    # Load GGUF tensors
    loader = VibeVoiceGGUFLoader()
    state_dict, config = loader.load_gguf_model(
        str(gguf_path),
        str(config_path) if config_path.exists() else None,
        keep_quantized=True
    )
    
    if not config:
        raise RuntimeError("GGUF model requires config.json")
    
    try:
        # Create minimal wrapper - no heavy computation
        print(f"🚀 Creating minimal wrapper (no full model)...")
        wrapper = MinimalVibeVoiceWrapper(state_dict, config, device)
        
        # Move to device
        wrapper = wrapper.to(device)
        wrapper.eval()
        
        print(f"✅ Direct GGUF loading complete - ready for lazy initialization")
        return wrapper, True
        
    except Exception as e:
        print(f"❌ Direct loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


class LazyVibeVoiceModel:
    """
    Lazy loading wrapper that creates the full model only when needed
    """
    def __init__(self, minimal_wrapper, original_config):
        self.minimal_wrapper = minimal_wrapper
        self.config = original_config
        self.full_model = None
        self.device = minimal_wrapper.device
        
    def _ensure_full_model(self):
        """Create full model on first use"""
        if self.full_model is None:
            print(f"🔄 Lazy loading: Creating full model on first use...")
            
            from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            
            # Now create the full model (this will be slow, but only once)
            model_config = VibeVoiceConfig.from_dict(self.config)
            self.full_model = VibeVoiceForConditionalGenerationInference(model_config)
            
            # Load the GGUF tensors with adaptation
            self._load_tensors_to_full_model()
            
            # Convert to GGUF ops
            from .gguf_ops import replace_linear_with_gguf
            replace_linear_with_gguf(self.full_model)
            
            self.full_model = self.full_model.to(self.device)
            self.full_model.eval()
            
            print(f"✅ Full model created and loaded")
    
    def _load_tensors_to_full_model(self):
        """Load tensors from minimal wrapper to full model with adaptation"""
        full_state = self.full_model.state_dict()
        wrapper_tensors = self.minimal_wrapper.tensor_storage
        
        adapted = 0
        loaded = 0
        
        for full_key in full_state.keys():
            # Try to find matching tensor in wrapper
            clean_key = full_key.replace('.', '_').replace('-', '_')
            
            if clean_key in wrapper_tensors:
                gguf_tensor = wrapper_tensors[clean_key]
                
                # Adapt dimensions if needed
                if full_state[full_key].shape == gguf_tensor.shape:
                    full_state[full_key] = gguf_tensor.data
                    loaded += 1
                else:
                    # Try simple adaptation
                    adapted_tensor = adapt_tensor_for_full_model(
                        gguf_tensor.data, 
                        full_state[full_key].shape
                    )
                    if adapted_tensor is not None:
                        full_state[full_key] = adapted_tensor
                        adapted += 1
        
        print(f"📊 Tensor loading: {loaded} direct, {adapted} adapted")
        self.full_model.load_state_dict(full_state, strict=False)
    
    def __call__(self, *args, **kwargs):
        """Forward pass - triggers lazy loading"""
        self._ensure_full_model()
        return self.full_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate method - triggers lazy loading"""
        self._ensure_full_model()
        return self.full_model.generate(*args, **kwargs)
    
    def eval(self):
        """Set to eval mode"""
        if self.full_model is not None:
            self.full_model.eval()
        return self
    
    def to(self, device):
        """Move to device"""
        self.device = device
        self.minimal_wrapper = self.minimal_wrapper.to(device)
        if self.full_model is not None:
            self.full_model = self.full_model.to(device)
        return self


def adapt_tensor_for_full_model(gguf_tensor, target_shape):
    """Simple tensor adaptation for full model loading"""
    from .gguf_utils import dequantize_tensor
    
    # Dequantize if needed
    if hasattr(gguf_tensor, 'tensor_type'):
        tensor = dequantize_tensor(gguf_tensor, dtype=torch.float16)
    else:
        tensor = gguf_tensor
    
    if len(tensor.shape) != len(target_shape):
        return None
    
    # Truncate or pad
    if all(t <= s for t, s in zip(target_shape, tensor.shape)):
        # Truncate
        slices = tuple(slice(0, t) for t in target_shape)
        return tensor[slices].contiguous()
    elif all(t >= s for t, s in zip(target_shape, tensor.shape)):
        # Pad
        pad_widths = []
        for t, s in zip(reversed(target_shape), reversed(tensor.shape)):
            pad_widths.extend([0, t - s])
        return torch.nn.functional.pad(tensor, pad_widths, mode='constant', value=0)
    
    return None


def create_lazy_gguf_model(model_path: str, device: torch.device) -> Tuple[Any, bool]:
    """
    Create lazy GGUF model that loads instantly and creates full model on first use
    """
    wrapper, success = load_gguf_direct(model_path, device)
    
    if success:
        # Create lazy wrapper
        lazy_model = LazyVibeVoiceModel(wrapper, wrapper.config)
        return lazy_model, True
    else:
        return None, False