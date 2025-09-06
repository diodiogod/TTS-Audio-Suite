"""
Python fallback implementation for monotonic alignment search
This replaces the typically compiled C extension with a pure Python version
Optimized with numba JIT compilation when available
"""

import numpy as np
import torch

# Try to use numba for speed optimization
try:
    from numba import jit
    HAS_NUMBA = True
    print("🚀 Using numba JIT for monotonic alignment search optimization")
except ImportError:
    HAS_NUMBA = False
    print("⚠️ Numba not available, using pure Python for monotonic alignment search")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def maximum_path_c(path, value, t_y, t_x):
    """
    Pure Python implementation of maximum path algorithm.
    This is a fallback for the typically compiled C extension.
    """
    b = path.shape[0]
    max_neg_val = np.finfo(np.float32).min
    
    for i in range(b):
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if y == 0:
                    v_prev = 0.0
                else:
                    v_prev = value[i, y - 1, x]
                    
                if x == 0:
                    v_cur = max_neg_val
                else:
                    v_cur = value[i, y, x - 1]
                
                if v_cur > v_prev:
                    value[i, y, x] += v_cur
                    path[i, y, x] = 1
                else:
                    value[i, y, x] += v_prev
                    path[i, y, x] = 0


def maximum_path(neg_x_ent, attn_mask):
    """
    Computes the maximum path through the attention matrix.
    
    Args:
        neg_x_ent: Negative cross-entropy tensor of shape (batch, t_y, t_x)
        attn_mask: Attention mask tensor of shape (batch, t_y, t_x)
        
    Returns:
        path: Maximum path tensor of shape (batch, t_y, t_x)
    """
    device = neg_x_ent.device
    dtype = neg_x_ent.dtype
    b, t_y, t_x = neg_x_ent.shape
    
    # Convert to numpy for processing
    neg_x_ent_np = neg_x_ent.detach().cpu().numpy()
    attn_mask_np = attn_mask.detach().cpu().numpy()
    
    # Initialize path and value matrices
    path = np.zeros((b, t_y, t_x), dtype=np.int32)
    value = neg_x_ent_np.copy()
    
    # Apply attention mask
    value = value * attn_mask_np
    
    # Compute maximum path
    for i in range(b):
        t_y_curr = int(attn_mask_np[i, :, 0].sum())
        t_x_curr = int(attn_mask_np[i, 0, :].sum())
        maximum_path_c(path, value, t_y_curr, t_x_curr)
    
    # Convert back to torch tensor
    return torch.from_numpy(path).to(device=device, dtype=torch.int32)


def maximum_path_jit(neg_x_ent, attn_mask):
    """
    JIT-compatible version of maximum path (fallback to regular version).
    """
    return maximum_path(neg_x_ent, attn_mask)