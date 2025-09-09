"""
TTS Audio Suite Compatibility Module
Centralized compatibility fixes for Python 3.13, transformers, numba, and other dependencies
"""

from .numba_compat import (
    setup_numba_compatibility,
    is_jit_disabled,
    get_compatibility_status,
    force_apply_workaround
)

from .transformers_patches import *

__all__ = [
    # Numba compatibility
    'setup_numba_compatibility',
    'is_jit_disabled', 
    'get_compatibility_status',
    'force_apply_workaround',
    
    # Transformers compatibility (from transformers_patches.py)
    # Add any exports from transformers_patches if needed
]