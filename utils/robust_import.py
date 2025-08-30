"""
Robust Import Helper for TTS Audio Suite
Ensures consistent module loading across all environments (conda, venv, Windows, etc.)
"""

import sys
import os
import importlib
from typing import Any

def ensure_project_path():
    """Ensure project root is in sys.path - works from any subdirectory"""
    # Find project root by looking for characteristic files
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # Walk up directories to find project root
    while current_path and current_path != os.path.dirname(current_path):  # Not at filesystem root
        # Check for project markers
        if (os.path.exists(os.path.join(current_path, "nodes.py")) and 
            os.path.exists(os.path.join(current_path, "utils")) and
            os.path.exists(os.path.join(current_path, "engines"))):
            # Found project root
            if current_path not in sys.path:
                sys.path.insert(0, current_path)
            return current_path
        current_path = os.path.dirname(current_path)
    
    # Fallback: assume we're in the project somewhere
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils/
    project_root = os.path.dirname(current_dir)  # project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

def robust_import(module_name: str, retry_count: int = 2) -> Any:
    """
    Robust import that handles path issues across all environments
    
    Args:
        module_name: Module to import (e.g., 'utils.audio.audio_hash')
        retry_count: Number of retry attempts with different strategies
        
    Returns:
        Imported module
        
    Raises:
        ImportError: If all strategies fail
    """
    # Strategy 1: Normal import
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as first_error:
        if retry_count <= 0:
            raise first_error
    
    # Strategy 2: Ensure path and invalidate cache
    try:
        ensure_project_path()
        importlib.invalidate_caches()
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as second_error:
        if retry_count <= 1:
            raise second_error
    
    # Strategy 3: Force reload of sys.modules if already partially loaded
    try:
        if module_name in sys.modules:
            # Remove from cache and retry
            del sys.modules[module_name]
            importlib.invalidate_caches()
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as final_error:
        raise ImportError(f"Failed to import '{module_name}' after all strategies. "
                         f"First error: {first_error}, Final error: {final_error}")

def robust_from_import(module_name: str, attr_names: list, retry_count: int = 2) -> dict:
    """
    Robust 'from module import attr' that handles path issues
    
    Args:
        module_name: Module to import from (e.g., 'utils.audio.audio_hash')
        attr_names: List of attribute names to import (e.g., ['generate_stable_audio_component'])
        retry_count: Number of retry attempts
        
    Returns:
        Dict mapping attribute names to imported objects
        
    Example:
        attrs = robust_from_import('utils.audio.audio_hash', ['generate_stable_audio_component'])
        generate_stable_audio_component = attrs['generate_stable_audio_component']
    """
    module = robust_import(module_name, retry_count)
    result = {}
    for attr_name in attr_names:
        if not hasattr(module, attr_name):
            raise ImportError(f"Module '{module_name}' has no attribute '{attr_name}'")
        result[attr_name] = getattr(module, attr_name)
    return result

# Initialize path on module load
ensure_project_path()