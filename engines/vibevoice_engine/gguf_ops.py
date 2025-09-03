"""
GGUF Operations for VibeVoice - Custom operations for quantized tensors
Handles on-the-fly dequantization for VRAM savings
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from .gguf_utils import is_quantized, dequantize_tensor

class GGUFLinear(nn.Module):
    """Linear layer that handles quantized weights"""
    
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Store quantized weight or regular weight
        if is_quantized(original_linear.weight):
            self.weight = original_linear.weight  # Keep quantized
            self._is_quantized = True
        else:
            self.weight = original_linear.weight  # Regular tensor
            self._is_quantized = False
            
        # Bias is usually not quantized
        self.bias = original_linear.bias
        
    def forward(self, x):
        if self._is_quantized:
            # Dequantize weight on-the-fly for computation
            weight_dequant = dequantize_tensor(self.weight, dtype=x.dtype)
        else:
            weight_dequant = self.weight
            
        return torch.nn.functional.linear(x, weight_dequant, self.bias)

def replace_linear_with_gguf(module: nn.Module, name: str = ""):
    """Recursively replace Linear layers with GGUF-aware versions"""
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            # Replace with GGUF-aware linear layer
            gguf_linear = GGUFLinear(child_module)
            setattr(module, child_name, gguf_linear)
        else:
            # Recursively process child modules
            replace_linear_with_gguf(child_module, f"{name}.{child_name}" if name else child_name)

def prepare_model_for_gguf(model: nn.Module) -> nn.Module:
    """Prepare a model to work with GGUF quantized tensors"""
    # Replace all Linear layers with GGUF-aware versions
    replace_linear_with_gguf(model)
    return model