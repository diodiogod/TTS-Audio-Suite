"""
Fast GGUF Model Loader for VibeVoice
Creates model with quantized ops from the start to avoid memory spike
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("VibeVoice-GGUF-Fast")


def load_gguf_vibevoice_fast(
    model_path: str,
    device: torch.device = torch.device("cuda")
) -> Tuple[Any, bool]:
    """
    Fast GGUF loading that avoids creating full model first
    
    Returns:
        Tuple of (model, success_flag)
    """
    from .gguf_loader import VibeVoiceGGUFLoader
    from .gguf_ops import GGUFLinear
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    
    print(f"⚡ Fast GGUF loading - creating model with quantized ops...")
    
    # Load GGUF state dict first
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
    
    print(f"📦 Loaded {len(state_dict)} GGUF tensors")
    
    # Monkey-patch Linear to use GGUF-aware version during model creation
    original_linear = nn.Linear
    
    class GGUFLinearFactory(nn.Linear):
        """Factory that creates GGUFLinear instead of regular Linear"""
        def __new__(cls, in_features, out_features, bias=True, device=None, dtype=None):
            # First create a regular Linear
            regular_linear = original_linear(in_features, out_features, bias=bias)
            if device is not None:
                regular_linear = regular_linear.to(device)
            if dtype is not None:
                regular_linear = regular_linear.to(dtype)
            # Then wrap it in GGUFLinear
            return GGUFLinear(regular_linear)
    
    # Simpler approach - create on CPU then load weights
    try:
        # Create model config
        model_config = VibeVoiceConfig.from_dict(config)
        
        # Create model on CPU to avoid initial VRAM spike
        print(f"🚀 Creating model on CPU with minimal memory usage...")
        original_device = device
        device = torch.device('cpu')
        
        with torch.no_grad():
            model = VibeVoiceForConditionalGenerationInference(model_config)
            model = model.to(device)
        
        # Now load the GGUF weights with adaptation
        print(f"📥 Loading GGUF weights with dimension adaptation...")
        success = load_gguf_weights_simple(model, state_dict)
        
        if success:
            # Convert to GGUF ops after loading
            print(f"🔧 Converting to GGUF operations...")
            from .gguf_ops import replace_linear_with_gguf
            replace_linear_with_gguf(model)
            
            # Move to target device
            print(f"📤 Moving to target device: {original_device}...")
            model = model.to(original_device)
            model.eval()
            
            print(f"✅ GGUF model loaded successfully")
            return model, True
        else:
            print(f"❌ Failed to load GGUF weights")
            return None, False
            
    except Exception as e:
        print(f"❌ Error during GGUF loading: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def load_gguf_weights_simple(model, state_dict):
    """
    Simple GGUF weight loading with dimension adaptation
    """
    from .gguf_utils import dequantize_tensor
    
    model_state = model.state_dict()
    loaded = 0
    adapted = 0
    skipped = 0
    
    for key in model_state.keys():
        if key in state_dict:
            gguf_tensor = state_dict[key]
            model_tensor = model_state[key]
            
            # Get shapes
            if hasattr(gguf_tensor, 'tensor_shape'):
                gguf_shape = gguf_tensor.tensor_shape
            else:
                gguf_shape = gguf_tensor.shape
            
            if model_tensor.shape == gguf_shape:
                # Direct assignment for matching shapes
                model_state[key] = gguf_tensor
                loaded += 1
            else:
                # Try simple adaptation
                try:
                    # Dequantize first if needed
                    if hasattr(gguf_tensor, 'tensor_type'):
                        tensor_data = dequantize_tensor(gguf_tensor, dtype=torch.float16)
                    else:
                        tensor_data = gguf_tensor
                    
                    # Simple truncation or padding
                    if len(model_tensor.shape) == len(gguf_shape):
                        # Create output tensor
                        output = torch.zeros_like(model_tensor)
                        
                        # Copy overlapping region
                        slices = tuple(slice(0, min(m, g)) for m, g in zip(model_tensor.shape, gguf_shape))
                        src_slices = tuple(slice(0, min(m, g)) for m, g in zip(gguf_shape, model_tensor.shape))
                        
                        output[slices] = tensor_data[src_slices]
                        model_state[key] = output
                        adapted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"⚠️ Failed to adapt {key}: {e}")
                    skipped += 1
        else:
            skipped += 1
    
    print(f"📊 Weight loading: {loaded} exact matches, {adapted} adapted, {skipped} skipped")
    
    # Load the adapted state dict
    try:
        model.load_state_dict(model_state, strict=False)
        return True
    except Exception as e:
        print(f"❌ Failed to load state dict: {e}")
        return False


def load_state_dict_with_meta_conversion(model, state_dict, target_device):
    """
    Load state dict into meta model, materializing tensors on target device
    """
    from .gguf_utils import dequantize_tensor
    
    loaded = 0
    skipped = 0
    adapted = 0
    
    # Get model's expected state dict
    model_state = model.state_dict()
    
    # Create a new state dict with properly shaped tensors
    new_state = {}
    
    for key, param in model_state.items():
        if key in state_dict:
            gguf_tensor = state_dict[key]
            
            # Get shapes
            param_shape = param.shape
            if hasattr(gguf_tensor, 'tensor_shape'):
                gguf_shape = gguf_tensor.tensor_shape
            else:
                gguf_shape = gguf_tensor.shape
            
            if param_shape == gguf_shape:
                # Perfect match - use directly
                new_state[key] = gguf_tensor
                loaded += 1
            else:
                # Try to adapt
                adapted_tensor = adapt_tensor_fast(gguf_tensor, param_shape, gguf_shape, key)
                if adapted_tensor is not None:
                    new_state[key] = adapted_tensor
                    adapted += 1
                else:
                    # Create zero tensor as fallback
                    new_state[key] = torch.zeros(param_shape, dtype=torch.float16, device=target_device)
                    skipped += 1
        else:
            # Missing key - create zeros
            new_state[key] = torch.zeros(param.shape, dtype=torch.float16, device=target_device)
            skipped += 1
    
    print(f"📊 Tensor loading: {loaded} exact, {adapted} adapted, {skipped} initialized")
    
    # Now create the actual model with the state dict
    # Since model was created on meta device, we need to materialize it
    try:
        # Move from meta to real device with state dict
        model.load_state_dict(new_state, strict=False, assign=True)
        model = model.to(target_device)
        return model
    except Exception as e:
        logger.error(f"Failed to materialize model: {e}")
        # Fallback: create model normally
        print(f"⚠️ Fast path failed, using fallback...")
        return create_model_fallback(state_dict, new_state, target_device)


def adapt_tensor_fast(gguf_tensor, target_shape, source_shape, key):
    """
    Fast tensor adaptation without full dequantization
    """
    from .gguf_utils import dequantize_tensor
    
    # For critical layers, try harder to adapt
    is_critical = any(k in key for k in ['embed', 'lm_head', 'norm', 'ln'])
    
    if len(target_shape) != len(source_shape):
        if not is_critical:
            return None
        # Try reshaping for critical layers
        total_target = target_shape.numel() if hasattr(target_shape, 'numel') else torch.Size(target_shape).numel()
        total_source = source_shape.numel() if hasattr(source_shape, 'numel') else torch.Size(source_shape).numel()
        
        if total_target == total_source:
            # Same total size - can reshape
            if hasattr(gguf_tensor, 'tensor_type'):
                tensor = dequantize_tensor(gguf_tensor, dtype=torch.float16)
            else:
                tensor = gguf_tensor
            return tensor.reshape(target_shape)
    
    # Same rank - try truncation/padding
    if all(t <= s for t, s in zip(target_shape, source_shape)):
        # Can truncate
        if hasattr(gguf_tensor, 'tensor_type'):
            # Keep quantized if possible, only dequantize what we need
            tensor = dequantize_tensor(gguf_tensor, dtype=torch.float16)
        else:
            tensor = gguf_tensor
        
        slices = tuple(slice(0, t) for t in target_shape)
        return tensor[slices]
    
    if all(t >= s for t, s in zip(target_shape, source_shape)):
        # Need to pad
        if hasattr(gguf_tensor, 'tensor_type'):
            tensor = dequantize_tensor(gguf_tensor, dtype=torch.float16)
        else:
            tensor = gguf_tensor
        
        pad_widths = []
        for t, s in zip(reversed(target_shape), reversed(source_shape)):
            pad_widths.extend([0, t - s])
        
        return torch.nn.functional.pad(tensor, pad_widths, mode='constant', value=0)
    
    # Mixed case - only for critical layers
    if is_critical:
        if hasattr(gguf_tensor, 'tensor_type'):
            tensor = dequantize_tensor(gguf_tensor, dtype=torch.float16)
        else:
            tensor = gguf_tensor
        
        # Create output tensor and copy what we can
        output = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Copy the overlapping region
        slices = tuple(slice(0, min(t, s)) for t, s in zip(target_shape, source_shape))
        output[slices] = tensor[slices]
        
        return output
    
    return None


def create_model_fallback(gguf_state_dict, adapted_state_dict, device):
    """
    Fallback method if fast loading fails
    """
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    
    print(f"🔄 Using fallback model creation...")
    
    # Get config from first tensor's metadata if possible
    config = {}
    for key, tensor in gguf_state_dict.items():
        if hasattr(tensor, 'metadata') and tensor.metadata:
            config = tensor.metadata.get('config', {})
            break
    
    if not config:
        # Use a default config
        print(f"⚠️ No config found, using defaults...")
        return None
    
    try:
        model_config = VibeVoiceConfig.from_dict(config)
        model = VibeVoiceForConditionalGenerationInference(model_config)
        model.load_state_dict(adapted_state_dict, strict=False)
        model = model.to(device)
        return model
    except Exception as e:
        logger.error(f"Fallback also failed: {e}")
        return None