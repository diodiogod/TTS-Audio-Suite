"""
GGUF Skeleton Loader - Creates minimal model structure without parameter allocation
Inspired by ComfyUI-GGUF approach but adapted for VibeVoice
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

def create_vibevoice_skeleton_from_gguf(
    model_path: str,
    device: torch.device
) -> Tuple[Any, bool]:
    """
    Create VibeVoice model skeleton and load GGUF weights directly
    This avoids creating the full parameter set first
    """
    from .gguf_loader import VibeVoiceGGUFLoader
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    
    print(f"🏗️ Creating VibeVoice skeleton for GGUF loading...")
    
    # Load GGUF data
    gguf_path = Path(model_path) / "model.gguf"
    config_path = Path(model_path) / "config.json"
    
    loader = VibeVoiceGGUFLoader()
    state_dict, config = loader.load_gguf_model(
        str(gguf_path),
        str(config_path) if config_path.exists() else None,
        keep_quantized=True
    )
    
    if not config:
        raise RuntimeError("GGUF model requires config.json")
    
    print(f"📊 GGUF contains {len(state_dict)} tensors")
    
    # Create model with custom initialization to avoid full parameter allocation
    try:
        model = create_skeleton_model(config, state_dict, device)
        if model is not None:
            model.eval()
            print(f"✅ GGUF VibeVoice skeleton loaded successfully")
            return model, True
        else:
            return None, False
            
    except Exception as e:
        print(f"❌ Skeleton creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def create_skeleton_model(config_dict, gguf_state_dict, device):
    """
    Create model with skeleton approach - allocate parameters only when needed
    """
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    
    # Patch parameter creation to be lazy
    original_param = nn.Parameter
    original_buffer = nn.Module.register_buffer
    
    class LazyParameter:
        """Placeholder for parameters that will be filled from GGUF"""
        def __init__(self, shape, dtype=torch.float32, device='cpu'):
            self.shape = torch.Size(shape)
            self.dtype = dtype
            self.device = device
            self.data = None  # Will be filled later
        
        def to(self, *args, **kwargs):
            return self
    
    def lazy_param_factory(data, requires_grad=True):
        """Create lazy parameters during model construction"""
        if isinstance(data, torch.Tensor):
            return LazyParameter(data.shape, data.dtype, data.device)
        return original_param(data, requires_grad)
    
    def lazy_buffer_factory(self, name, tensor, persistent=True):
        """Create lazy buffers during model construction"""
        if isinstance(tensor, torch.Tensor):
            setattr(self, name, LazyParameter(tensor.shape, tensor.dtype, tensor.device))
        else:
            original_buffer(self, name, tensor, persistent)
    
    try:
        # Temporarily replace parameter creation
        nn.Parameter = lazy_param_factory
        nn.Module.register_buffer = lazy_buffer_factory
        
        # Create model config
        model_config = VibeVoiceConfig.from_dict(config_dict)
        
        print(f"🔨 Building model skeleton...")
        # Create model structure without allocating full parameters
        with torch.no_grad():
            model = VibeVoiceForConditionalGenerationInference(model_config)
        
        print(f"💉 Injecting GGUF tensors into skeleton...")
        # Now replace lazy parameters with actual GGUF tensors
        inject_gguf_tensors(model, gguf_state_dict)
        
        # Move to target device
        print(f"📤 Moving to device: {device}")
        model = model.to(device)
        
        # Convert Linear layers to GGUF-aware versions
        from .gguf_ops import replace_linear_with_gguf
        print(f"🔧 Converting to GGUF operations...")
        replace_linear_with_gguf(model)
        
        return model
        
    finally:
        # Restore original functions
        nn.Parameter = original_param
        nn.Module.register_buffer = original_buffer


def inject_gguf_tensors(model, gguf_state_dict):
    """
    Replace lazy parameters with actual GGUF tensors
    """
    from .gguf_utils import dequantize_tensor
    
    injected = 0
    adapted = 0
    missing = 0
    
    def inject_recursive(module, prefix=""):
        nonlocal injected, adapted, missing
        
        # Process parameters
        for name, param in list(module.named_parameters(recurse=False)):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if full_name in gguf_state_dict:
                gguf_tensor = gguf_state_dict[full_name]
                
                # Get expected shape
                if hasattr(param, 'shape'):
                    expected_shape = param.shape
                elif isinstance(param, LazyParameter):
                    expected_shape = param.shape
                else:
                    continue
                
                # Get GGUF shape
                if hasattr(gguf_tensor, 'tensor_shape'):
                    gguf_shape = gguf_tensor.tensor_shape
                else:
                    gguf_shape = gguf_tensor.shape
                
                if expected_shape == gguf_shape:
                    # Direct replacement
                    setattr(module, name, nn.Parameter(gguf_tensor, requires_grad=False))
                    injected += 1
                else:
                    # Try adaptation
                    try:
                        # Dequantize if needed
                        if hasattr(gguf_tensor, 'tensor_type'):
                            tensor_data = dequantize_tensor(gguf_tensor, dtype=torch.float16)
                        else:
                            tensor_data = gguf_tensor
                        
                        # Adapt dimensions
                        adapted_tensor = adapt_tensor_simple(tensor_data, expected_shape, gguf_shape)
                        if adapted_tensor is not None:
                            setattr(module, name, nn.Parameter(adapted_tensor, requires_grad=False))
                            adapted += 1
                        else:
                            # Create zero tensor as fallback
                            zero_tensor = torch.zeros(expected_shape, dtype=torch.float16)
                            setattr(module, name, nn.Parameter(zero_tensor, requires_grad=False))
                            missing += 1
                    except Exception as e:
                        print(f"⚠️ Failed to inject {full_name}: {e}")
                        zero_tensor = torch.zeros(expected_shape, dtype=torch.float16)
                        setattr(module, name, nn.Parameter(zero_tensor, requires_grad=False))
                        missing += 1
            else:
                # Missing tensor - create zero
                if hasattr(param, 'shape'):
                    zero_tensor = torch.zeros(param.shape, dtype=torch.float16)
                    setattr(module, name, nn.Parameter(zero_tensor, requires_grad=False))
                    missing += 1
        
        # Process buffers
        for name, buffer in list(module.named_buffers(recurse=False)):
            full_name = f"{prefix}.{name}" if prefix else name
            
            if full_name in gguf_state_dict:
                gguf_tensor = gguf_state_dict[full_name]
                module.register_buffer(name, gguf_tensor.detach() if hasattr(gguf_tensor, 'detach') else gguf_tensor)
                injected += 1
        
        # Recurse to children
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            inject_recursive(child, child_prefix)
    
    inject_recursive(model)
    print(f"💉 Tensor injection: {injected} direct, {adapted} adapted, {missing} missing/zero")


def adapt_tensor_simple(tensor, target_shape, source_shape):
    """
    Simple tensor adaptation - truncate or pad as needed
    """
    if len(target_shape) != len(source_shape):
        return None
    
    # Check if we can truncate or need to pad
    can_truncate = all(t <= s for t, s in zip(target_shape, source_shape))
    need_pad = all(t >= s for t, s in zip(target_shape, source_shape))
    
    if can_truncate:
        # Truncate to fit
        slices = tuple(slice(0, t) for t in target_shape)
        return tensor[slices].contiguous()
    
    if need_pad:
        # Pad to fit
        pad_widths = []
        for t, s in zip(reversed(target_shape), reversed(source_shape)):
            pad_widths.extend([0, t - s])
        return torch.nn.functional.pad(tensor, pad_widths, mode='constant', value=0)
    
    # Mixed case - create new tensor and copy what fits
    output = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Copy overlapping region
    min_shape = tuple(min(t, s) for t, s in zip(target_shape, source_shape))
    slices = tuple(slice(0, m) for m in min_shape)
    
    output[slices] = tensor[slices]
    return output


class LazyParameter:
    """Placeholder for parameters that will be filled from GGUF"""
    def __init__(self, shape, dtype=torch.float32, device='cpu'):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = device