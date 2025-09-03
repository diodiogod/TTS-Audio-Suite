"""
GGUF Model Loader for VibeVoice
Handles loading GGUF models with dimension adaptation and quantization support
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, Any

def load_gguf_model_for_vibevoice(
    model_path: str,
    model_class: Any,
    config_dict: dict,
    device: torch.device
) -> Tuple[Any, bool]:
    """
    Load a GGUF model for VibeVoice with proper dimension handling
    
    Returns:
        Tuple of (model, success_flag)
    """
    from .gguf_loader import VibeVoiceGGUFLoader
    from .gguf_utils import dequantize_tensor
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    
    print(f"📋 Loading GGUF model with dimension adaptation...")
    
    # Load GGUF file
    gguf_path = Path(model_path) / "model.gguf"
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
    
    # Load config path
    config_path = Path(model_path) / "config.json"
    
    # Use the loader class
    loader = VibeVoiceGGUFLoader()
    state_dict, config = loader.load_gguf_model(
        str(gguf_path), 
        str(config_path) if config_path.exists() else None,
        keep_quantized=True
    )
    
    # Use provided config if GGUF didn't have one
    if not config:
        config = config_dict
    
    # Create model
    print(f"🔧 Creating model structure on CPU...")
    model_config = VibeVoiceConfig.from_dict(config)
    
    with torch.no_grad():
        model = model_class(model_config)
        model = model.to(torch.device('cpu'))
    
    # Load and adapt state dict
    success = load_gguf_state_dict_with_adaptation(model, state_dict)
    
    if success:
        # Convert to GGUF-aware operations
        from .gguf_ops import replace_linear_with_gguf
        print(f"🔧 Converting Linear layers for GGUF quantized inference...")
        replace_linear_with_gguf(model)
        
        # Move to target device
        print(f"🔧 Moving model to target device: {device}...")
        model = model.to(device)
        model.eval()
        print(f"✅ GGUF VibeVoice model loaded successfully with VRAM optimization")
        return model, True
    else:
        print(f"❌ Failed to load GGUF model with adaptation")
        return None, False


def load_gguf_state_dict_with_adaptation(model: torch.nn.Module, state_dict: Dict) -> bool:
    """
    Load GGUF state dict with dimension adaptation
    
    Returns:
        True if successful, False otherwise
    """
    from .gguf_utils import dequantize_tensor
    
    model_state = model.state_dict()
    loaded_keys = []
    adapted_keys = []
    skipped_keys = []
    critical_failures = []
    
    for key, gguf_tensor in state_dict.items():
        if key not in model_state:
            continue
            
        model_tensor = model_state[key]
        
        # Get GGUF tensor shape
        if hasattr(gguf_tensor, 'tensor_shape'):
            gguf_shape = gguf_tensor.tensor_shape
        else:
            gguf_shape = gguf_tensor.shape
        
        if model_tensor.shape == gguf_shape:
            # Perfect match
            model_state[key] = gguf_tensor
            loaded_keys.append(key)
        else:
            # Try to adapt dimensions
            adapted = adapt_tensor_dimensions(
                gguf_tensor, 
                model_tensor.shape, 
                gguf_shape,
                key
            )
            
            if adapted is not None:
                model_state[key] = adapted
                adapted_keys.append(key)
            else:
                skipped_keys.append(key)
                # Check if this is a critical layer
                if any(critical in key for critical in ['embed', 'lm_head', 'acoustic_tokenizer']):
                    critical_failures.append(key)
    
    print(f"📦 GGUF loading summary:")
    print(f"   ✅ Exact matches: {len(loaded_keys)}")
    print(f"   🔧 Adapted: {len(adapted_keys)}")
    print(f"   ⚠️ Skipped: {len(skipped_keys)}")
    
    if critical_failures:
        print(f"   ❌ Critical layers failed: {critical_failures[:3]}")
        return False
    
    # Load the adapted state dict
    try:
        model.load_state_dict(model_state, strict=False)
        return True
    except Exception as e:
        print(f"❌ Failed to load adapted state dict: {e}")
        return False


def adapt_tensor_dimensions(
    gguf_tensor: Any,
    target_shape: torch.Size,
    source_shape: torch.Size,
    key: str
) -> Optional[torch.Tensor]:
    """
    Adapt GGUF tensor dimensions to match model expectations
    
    Returns:
        Adapted tensor or None if adaptation not possible
    """
    from .gguf_utils import dequantize_tensor
    
    # First dequantize if needed
    if hasattr(gguf_tensor, 'tensor_type'):
        tensor = dequantize_tensor(gguf_tensor)
    else:
        tensor = gguf_tensor
    
    # Same number of dimensions - try adaptation
    if len(target_shape) == len(source_shape):
        return adapt_same_rank_tensors(tensor, target_shape, source_shape, key)
    
    # Different number of dimensions - more complex
    return adapt_different_rank_tensors(tensor, target_shape, source_shape, key)


def adapt_same_rank_tensors(
    tensor: torch.Tensor,
    target_shape: torch.Size,
    source_shape: torch.Size,
    key: str
) -> Optional[torch.Tensor]:
    """Adapt tensors with same number of dimensions"""
    
    # All dimensions smaller - pad
    if all(t >= s for t, s in zip(target_shape, source_shape)):
        return pad_tensor(tensor, target_shape)
    
    # All dimensions larger - truncate
    if all(t <= s for t, s in zip(target_shape, source_shape)):
        return truncate_tensor(tensor, target_shape)
    
    # Mixed - try smart adaptation for specific layer types
    if 'weight' in key:
        return adapt_weight_tensor(tensor, target_shape, source_shape, key)
    
    return None


def adapt_different_rank_tensors(
    tensor: torch.Tensor,
    target_shape: torch.Size,
    source_shape: torch.Size,
    key: str
) -> Optional[torch.Tensor]:
    """Adapt tensors with different number of dimensions"""
    
    # Add dimensions if needed
    if len(target_shape) > len(source_shape):
        # Add singleton dimensions
        for _ in range(len(target_shape) - len(source_shape)):
            tensor = tensor.unsqueeze(-1)
        return adapt_same_rank_tensors(tensor, target_shape, tensor.shape, key)
    
    # Remove dimensions if possible
    if len(target_shape) < len(source_shape):
        # Try squeezing singleton dimensions
        squeezed = tensor
        for dim in reversed(range(len(source_shape))):
            if source_shape[dim] == 1 and squeezed.ndim > len(target_shape):
                squeezed = squeezed.squeeze(dim)
        
        if squeezed.ndim == len(target_shape):
            return adapt_same_rank_tensors(squeezed, target_shape, squeezed.shape, key)
    
    return None


def pad_tensor(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Pad tensor to target shape"""
    pad_widths = []
    for t, s in zip(reversed(target_shape), reversed(tensor.shape)):
        pad_widths.extend([0, t - s])
    
    return F.pad(tensor, pad_widths, mode='constant', value=0)


def truncate_tensor(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Truncate tensor to target shape"""
    slices = tuple(slice(0, t) for t in target_shape)
    return tensor[slices]


def adapt_weight_tensor(
    tensor: torch.Tensor,
    target_shape: torch.Size,
    source_shape: torch.Size,
    key: str
) -> Optional[torch.Tensor]:
    """
    Smart adaptation for weight tensors
    Handles common patterns like Linear layer dimension changes
    """
    
    # For Linear layers (2D weights)
    if len(target_shape) == 2 and len(source_shape) == 2:
        out_features_t, in_features_t = target_shape
        out_features_s, in_features_s = source_shape
        
        # Create adapted tensor
        adapted = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Copy the overlapping region
        min_out = min(out_features_t, out_features_s)
        min_in = min(in_features_t, in_features_s)
        adapted[:min_out, :min_in] = tensor[:min_out, :min_in]
        
        # Initialize new weights if expanded
        if out_features_t > out_features_s or in_features_t > in_features_s:
            # Use small random values for new connections
            if out_features_t > out_features_s:
                adapted[out_features_s:, :min_in] = torch.randn(
                    out_features_t - out_features_s, min_in,
                    dtype=tensor.dtype, device=tensor.device
                ) * 0.01
            
            if in_features_t > in_features_s:
                adapted[:min_out, in_features_s:] = torch.randn(
                    min_out, in_features_t - in_features_s,
                    dtype=tensor.dtype, device=tensor.device
                ) * 0.01
        
        return adapted
    
    return None