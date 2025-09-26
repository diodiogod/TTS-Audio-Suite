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


def create_sage_sdpa_wrapper():
    """Create a SageAttention wrapper for F.scaled_dot_product_attention"""
    try:
        from sageattention import sageattn
        import torch.nn.functional as F

        # Store original function
        original_sdpa = F.scaled_dot_product_attention

        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            """Wrapper that replaces scaled_dot_product_attention with SageAttention"""
            # Log any unexpected parameters for debugging
            if kwargs:
                unexpected_params = list(kwargs.keys())
                logger.debug(f"SageAttention: Ignoring unsupported parameters: {unexpected_params}")

            try:
                # SageAttention expects tensors in specific format
                # Transformers typically use (batch, heads, seq_len, head_dim)

                # Check tensor dimensions to determine layout
                if query.dim() == 4:
                    # 4D tensor: (batch, heads, seq, dim)
                    batch_size, num_heads, seq_len_q, head_dim = query.shape
                    seq_len_k = key.shape[2]

                    # Reshape to (batch*heads, seq, dim) for HND layout
                    query_reshaped = query.reshape(batch_size * num_heads, seq_len_q, head_dim)
                    key_reshaped = key.reshape(batch_size * num_heads, seq_len_k, head_dim)
                    value_reshaped = value.reshape(batch_size * num_heads, seq_len_k, head_dim)

                    # Call sageattn with HND layout
                    output = sageattn(
                        query_reshaped, key_reshaped, value_reshaped,
                        is_causal=is_causal,
                        tensor_layout="HND"  # Heads*batch, seqN, Dim
                    )

                    # Reshape back to original format if needed
                    if output.dim() == 3:
                        output = output.reshape(batch_size, num_heads, seq_len_q, head_dim)

                    return output
                else:
                    # For 3D tensors, assume they're already in HND format
                    output = sageattn(
                        query, key, value,
                        is_causal=is_causal,
                        tensor_layout="HND"
                    )
                    return output

            except Exception as e:
                # If SageAttention fails, fall back to original implementation
                logger.debug(f"SageAttention failed, using original: {e}")
                # Call with proper arguments - scale is a keyword argument in PyTorch 2.0+
                if scale is not None:
                    return original_sdpa(query, key, value, attn_mask=attn_mask,
                                       dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
                else:
                    return original_sdpa(query, key, value, attn_mask=attn_mask,
                                       dropout_p=dropout_p, is_causal=is_causal, **kwargs)

        return sage_sdpa, original_sdpa

    except ImportError:
        logger.error("SageAttention not available for F.scaled_dot_product_attention wrapper")
        return None, None


def set_sage_attention(model):
    """
    Apply SageAttention using Enemyx's safer SDPA wrapper approach.
    This patches F.scaled_dot_product_attention instead of model layers.

    Args:
        model: The VibeVoice model to patch

    Raises:
        ImportError: If SageAttention is not installed
    """
    if not SAGE_ATTENTION_AVAILABLE:
        raise ImportError("SageAttention library is not installed or failed to load. Install with: pip install sageattention")

    # Use Enemyx's safer approach - patch F.scaled_dot_product_attention
    sage_sdpa, original_sdpa = create_sage_sdpa_wrapper()
    if sage_sdpa is None:
        logger.warning("Failed to create SageAttention SDPA wrapper")
        return

    # Apply Enemyx-style patching to attention modules
    import torch.nn.functional as F
    patched_count = 0

    def patch_attention_forward(module):
        """Patch attention layers to use SageAttention via SDPA wrapper"""
        nonlocal patched_count

        if hasattr(module, 'forward'):
            original_forward = module.forward

            def sage_forward(*args, **kwargs):
                """Wrapper that temporarily replaces F.scaled_dot_product_attention"""
                # Store and replace the function
                original_func = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = sage_sdpa

                try:
                    # Call original forward with patched attention
                    result = original_forward(*args, **kwargs)
                finally:
                    # Always restore original function
                    F.scaled_dot_product_attention = original_func

                return result

            # Check if this module likely uses attention
            module_name = module.__class__.__name__.lower()
            if any(name in module_name for name in ['attention', 'attn', 'multihead']):
                # Store original for restoration
                if not hasattr(module, '_original_forward'):
                    module._original_forward = original_forward
                module.forward = sage_forward
                patched_count += 1

        # Recursively patch child modules
        for child in module.children():
            patch_attention_forward(child)

    # Apply patching to the entire model
    patch_attention_forward(model)

    if patched_count > 0:
        logger.info(f"Patched {patched_count} attention layers with SageAttention (Enemyx method)")
    else:
        logger.warning("No attention layers found to patch - SageAttention may not be applied")


def restore_original_attention(model):
    """
    Restore original attention implementation by removing SageAttention patches.

    Args:
        model: The VibeVoice model to restore
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    except ImportError:
        logger.error("Qwen2Attention not found in transformers library")
        return

    restored_count = 0
    for module in model.modules():
        if isinstance(module, Qwen2Attention) and hasattr(module, '_original_forward'):
            module.forward = module._original_forward
            delattr(module, '_original_forward')
            restored_count += 1

    if restored_count > 0:
        logger.info(f"Restored original attention for {restored_count} Qwen2Attention layers")
    else:
        logger.warning("No patched Qwen2Attention layers found to restore")