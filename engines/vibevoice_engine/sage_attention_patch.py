"""
SageAttention implementation for VibeVoice models
Provides high-performance mixed-precision attention with GPU-specific optimizations
Based on implementation from wildminder/ComfyUI-VibeVoice
"""

import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False


def get_sage_attention_function_and_params():
    """
    Selects the best available SageAttention CUDA kernel and its parameters
    based on the current GPU architecture.
    
    Returns:
        tuple: (attention_function, quantization_granularity, accumulation_dtype)
               or (None, None, None) if not available
    """
    if not SAGE_ATTENTION_AVAILABLE or not torch.cuda.is_available():
        return None, None, None

    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor
    
    attn_func = None
    pv_accum_dtype = "fp32"

    if arch_code >= 120:  # Blackwell architecture (RTX 50 series)
        # Use same kernel as Ada for compatibility (fixes inappropriate assert on Blackwell)
        pv_accum_dtype = "fp32+fp32" 
        attn_func = sageattn_qk_int8_pv_fp8_cuda  # Same as SM89 Ada
        logger.info(f"SageAttention: Using SM120+ (Blackwell) with Ada kernel - pv_accum_dtype='{pv_accum_dtype}'.")
    elif arch_code >= 90:  # Hopper architecture (H100, etc.)
        pv_accum_dtype = "fp32+fp32" 
        attn_func = sageattn_qk_int8_pv_fp8_cuda_sm90
        logger.info(f"SageAttention: Using SM90+ (Hopper) FP8 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    elif arch_code == 89:  # Ada Lovelace (RTX 40 series)
        pv_accum_dtype = "fp32+fp32" 
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        logger.info(f"SageAttention: Using SM89 (Ada) FP8 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    elif arch_code >= 80:  # Ampere (RTX 30 series, A100)
        pv_accum_dtype = "fp32" 
        attn_func = sageattn_qk_int8_pv_fp16_cuda
        logger.info(f"SageAttention: Using SM80+ (Ampere) FP16 kernel with pv_accum_dtype='{pv_accum_dtype}'.")
    else:
        logger.warning(f"SageAttention not supported on current GPU architecture (SM{arch_code}).")
        return None, None, None
    
    return attn_func, "per_warp", pv_accum_dtype

# Initialize on module load
SAGE_ATTENTION_FUNCTION, QK_QUANT_GRAN, PV_ACCUM_DTYPE = get_sage_attention_function_and_params()


def sage_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional['Cache'] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Custom forward pass for Qwen2Attention using SageAttention kernels.
    Provides mixed-precision INT8/FP16/FP8 computation for improved performance.
    """
    
    if SAGE_ATTENTION_FUNCTION is None:
        raise RuntimeError("SageAttention was selected but no compatible kernel was found for this GPU.")
    
    original_dtype = hidden_states.dtype
    
    # Check if model is 4-bit quantized
    is_4bit = hasattr(self.q_proj, 'quant_state')
    if is_4bit:
        target_dtype = torch.bfloat16
    else:
        target_dtype = self.q_proj.weight.dtype

    if hidden_states.dtype != target_dtype:
        hidden_states = hidden_states.to(target_dtype)

    bsz, q_len, _ = hidden_states.size()

    # Linear projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape for attention computation
    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Apply rotary position embeddings
    cos, sin = position_embeddings
    # Import apply_rotary_pos_emb dynamically to avoid import errors
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids=None)

    # Handle past key values for caching
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Note: SageAttention kernel handles KV head broadcasting internally
    # No need to repeat K and V heads here

    is_causal = attention_mask is None and q_len > 1
    
    # Call SageAttention kernel
    attn_output = SAGE_ATTENTION_FUNCTION(
        query_states.to(target_dtype),
        key_states.to(target_dtype),
        value_states.to(target_dtype),
        tensor_layout="HND",  # Head, Number, Dimension layout
        is_causal=is_causal,
        qk_quant_gran=QK_QUANT_GRAN,
        pv_accum_dtype=PV_ACCUM_DTYPE,
    )
    
    # Handle potential tuple return
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0] 

    # Reshape output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
    
    # Output projection
    attn_output = self.o_proj(attn_output)

    # Restore original dtype if needed
    if attn_output.dtype != original_dtype:
        attn_output = attn_output.to(original_dtype)

    # SageAttention doesn't return attention weights
    attn_weights = None
    
    return attn_output, attn_weights


def set_sage_attention(model):
    """
    Apply SageAttention to all Qwen2Attention layers in the model.
    This monkey-patches the forward method to use SageAttention kernels.
    
    Args:
        model: The VibeVoice model to patch
        
    Raises:
        ImportError: If SageAttention is not installed
    """
    if not SAGE_ATTENTION_AVAILABLE:
        raise ImportError("SageAttention library is not installed or failed to load. Install with: pip install sageattention")
    
    if SAGE_ATTENTION_FUNCTION is None:
        logger.warning("SageAttention not available for current GPU architecture")
        return

    # Import Qwen2Attention dynamically
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    except ImportError:
        logger.error("Qwen2Attention not found in transformers library")
        return

    patched_count = 0
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module.forward = sage_attention_forward.__get__(module, Qwen2Attention)
            patched_count += 1
    
    if patched_count > 0:
        logger.info(f"Successfully patched {patched_count} Qwen2Attention layers with SageAttention")
    else:
        logger.warning("No Qwen2Attention layers found to patch")