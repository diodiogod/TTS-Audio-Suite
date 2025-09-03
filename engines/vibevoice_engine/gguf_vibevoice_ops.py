"""
VibeVoice GGUF Operations - Based on ComfyUI-GGUF approach
Creates model with custom ops that handle GGUF tensors directly
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

# Import ComfyUI operations
import comfy.ops

class GGUFLayer(nn.Module):
    """Base layer for GGUF operations, similar to ComfyUI-GGUF's GGMLLayer"""
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Custom loading from GGUF state dict"""
        weight_key = f"{prefix}weight"
        bias_key = f"{prefix}bias"
        
        # Load weight
        if weight_key in state_dict:
            gguf_tensor = state_dict[weight_key]
            # Store GGUF tensor as a regular attribute (not parameter)
            # and create a dummy Parameter for PyTorch compatibility
            super().__setattr__('_gguf_weight', gguf_tensor)
            # Create minimal dummy parameter based on tensor shape (don't dequantize!)
            if hasattr(gguf_tensor, 'tensor_shape'):
                dummy_shape = gguf_tensor.tensor_shape
            else:
                dummy_shape = gguf_tensor.shape
            super().__setattr__('weight', nn.Parameter(torch.zeros(dummy_shape, dtype=torch.float16), requires_grad=False))
        elif hasattr(self, 'in_features') and hasattr(self, 'out_features'):
            # Create zero weight for missing Linear weights
            self.weight = nn.Parameter(
                torch.zeros(self.out_features, self.in_features), 
                requires_grad=False
            )
            missing_keys.append(weight_key)
        
        # Load bias
        if bias_key in state_dict and state_dict[bias_key] is not None:
            gguf_bias = state_dict[bias_key]
            # Store GGUF tensor as a regular attribute (not parameter)
            # and create a dummy Parameter for PyTorch compatibility
            super().__setattr__('_gguf_bias', gguf_bias)
            # Create minimal dummy parameter based on tensor shape (don't dequantize!)
            if hasattr(gguf_bias, 'tensor_shape'):
                dummy_shape = gguf_bias.tensor_shape
            else:
                dummy_shape = gguf_bias.shape
            super().__setattr__('bias', nn.Parameter(torch.zeros(dummy_shape, dtype=torch.float16), requires_grad=False))
        elif hasattr(self, 'out_features'):
            # Handle missing bias
            if self.bias is not None:  # If bias was expected
                self.bias = nn.Parameter(
                    torch.zeros(self.out_features),
                    requires_grad=False
                )
                missing_keys.append(bias_key)


class VibeVoiceGGUFOps:
    """
    Custom operations for VibeVoice GGUF models
    Similar to ComfyUI-GGUF's GGMLOps but for VibeVoice
    """
    
    class Linear(GGUFLayer, comfy.ops.manual_cast.Linear):
        """
        Linear layer that can handle GGUF tensors
        Uses ComfyUI's native manual_cast.Linear like the reference implementation
        """
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # Initialize ComfyUI's Linear first
            comfy.ops.manual_cast.Linear.__init__(self, in_features, out_features, bias, device, dtype)
            # Initialize GGUFLayer
            GGUFLayer.__init__(self)
            
            # Ensure weight and bias attributes exist for Transformers compatibility
            if not hasattr(self, 'weight') or self.weight is None:
                self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
            
            if bias and (not hasattr(self, 'bias') or self.bias is None):
                self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
            elif not bias:
                self.bias = None
        
        def forward(self, input):
            from .gguf_utils import GGMLTensor, is_quantized
            
            # Real GGUF approach - dequantize on-demand, clear cache immediately
            weight_dequant = None
            bias_dequant = None
            
            # Handle GGUF weight
            if hasattr(self, '_gguf_weight') and isinstance(self._gguf_weight, GGMLTensor):
                weight_dequant = self._gguf_weight.dequantize_for_computation()
            elif hasattr(self, 'weight'):
                weight_dequant = self.weight
            else:
                raise RuntimeError("No weight tensor available")
            
            # Handle GGUF bias  
            if hasattr(self, '_gguf_bias') and isinstance(self._gguf_bias, GGMLTensor):
                bias_dequant = self._gguf_bias.dequantize_for_computation()
            elif hasattr(self, 'bias'):
                bias_dequant = self.bias
            else:
                bias_dequant = None
            
            # Ensure dtype consistency before computation
            if bias_dequant is not None and weight_dequant.dtype != bias_dequant.dtype:
                bias_dequant = bias_dequant.to(weight_dequant.dtype)
            
            # Perform computation
            result = torch.nn.functional.linear(input, weight_dequant, bias_dequant)
            
            # Clear caches immediately to save VRAM
            if hasattr(self, '_gguf_weight') and isinstance(self._gguf_weight, GGMLTensor):
                self._gguf_weight.clear_dequant_cache()
            if hasattr(self, '_gguf_bias') and isinstance(self._gguf_bias, GGMLTensor):
                self._gguf_bias.clear_dequant_cache()
            
            return result


class GGMLOps:
    """GGML Operations class for VibeVoice GGUF support"""
    
    def __init__(self):
        self.Linear = VibeVoiceGGUFOps.Linear


def create_vibevoice_with_gguf_ops(model_path: str, device: torch.device) -> Tuple[Any, bool]:
    """
    Create VibeVoice model with GGUF operations from the start
    This avoids the parameter creation issues
    """
    from .gguf_loader import VibeVoiceGGUFLoader
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    
    print(f"⚡ Creating VibeVoice with GGUF ops (ComfyUI-GGUF approach)...")
    
    # Load GGUF data first
    gguf_path = Path(model_path) / "model.gguf"
    config_path = Path(model_path) / "config.json"
    
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
    
    loader = VibeVoiceGGUFLoader()
    state_dict, config = loader.load_gguf_model(
        str(gguf_path),
        str(config_path) if config_path.exists() else None,
        keep_quantized=True
    )
    
    if not config:
        raise RuntimeError("GGUF model requires config.json")
    
    print(f"📦 Loaded {len(state_dict)} GGUF tensors")
    
    try:
        # Replace nn.Linear with our GGUF-aware version during model creation
        original_linear = nn.Linear
        nn.Linear = VibeVoiceGGUFOps.Linear
        
        try:
            # Create model config
            model_config = VibeVoiceConfig.from_dict(config)
            
            # Create model with GGUF ops
            print(f"🏗️ Creating VibeVoice model with GGUF operations...")
            model = VibeVoiceForConditionalGenerationInference(model_config)
            
            # Load GGUF state dict
            print(f"📥 Loading GGUF weights...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"📊 State dict loaded: {len(missing)} missing, {len(unexpected)} unexpected")
            
            # Move to device
            print(f"📤 Moving to device: {device}")
            model = model.to(device)
            model.eval()
            
            print(f"✅ VibeVoice GGUF model ready")
            return model, True
            
        finally:
            # Restore original Linear
            nn.Linear = original_linear
    
    except Exception as e:
        print(f"❌ GGUF ops model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def monkey_patch_vibevoice_for_gguf():
    """
    Monkey patch VibeVoice components to use GGUF-aware operations
    """
    # This would be more complex for a full implementation
    # For now, we use the simpler approach above
    pass