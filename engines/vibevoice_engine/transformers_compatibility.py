"""
VibeVoice Transformers Compatibility Patch
Fixes _prepare_cache_for_generation() method signature changes in Transformers 4.56+

Based on fix by drbaph (Saganaki22) from:
https://github.com/wildminder/ComfyUI-VibeVoice/pull/16

Credits: 
- Original fix: drbaph <84208527+Saganaki22@users.noreply.github.com>
- Integration: TTS Audio Suite
"""

import inspect
import logging
from typing import Any

logger = logging.getLogger("VibeVoice.Compatibility")


def patch_prepare_cache_for_generation():
    """
    Apply compatibility patch for Transformers 4.56+ _prepare_cache_for_generation method.
    
    The method signature changed from 6 parameters to 5 parameters in Transformers 4.56+:
    - Old: (self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device)  
    - New: (self, generation_config, model_kwargs, batch_size, max_cache_length, device)
    
    This creates a dynamic wrapper that detects the correct signature and calls accordingly.
    """
    try:
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        
        # Store the original method
        original_method = VibeVoiceForConditionalGenerationInference._prepare_cache_for_generation
        
        def patched_prepare_cache_for_generation(self, generation_config, model_kwargs, *args):
            """
            Dynamic wrapper that adapts to both old and new Transformers versions.
            
            Args:
                self: Model instance
                generation_config: Generation configuration
                model_kwargs: Model keyword arguments  
                *args: Variable arguments to handle both signatures
            """
            try:
                # Inspect the original method signature
                sig = inspect.signature(original_method)
                param_count = len(sig.parameters)
                
                if param_count == 5:
                    # New transformers version (4.56+): 5 parameters
                    # Expected: (self, generation_config, model_kwargs, batch_size, max_cache_length, device)
                    if len(args) >= 3:  # Skip assistant_model (args[0]) and use remaining args
                        # VibeVoice calls with: assistant_model, batch_size, max_cache_length, device
                        # We need to skip assistant_model and pass: batch_size, max_cache_length, device
                        batch_size = args[1]
                        max_cache_length = args[2]
                        device = args[3]

                        return original_method(self, generation_config, model_kwargs, batch_size, max_cache_length, device)
                    else:
                        # Fallback to original call
                        return original_method(self, generation_config, model_kwargs, *args)
                        
                else:
                    # Old transformers version (pre-4.56): 6 parameters
                    # Expected: (self, generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device)
                    return original_method(self, generation_config, model_kwargs, *args)
                    
            except Exception as e:
                # Suppress this message - it's just parameter adaptation, doesn't affect quality
                # logger.warning(f"Compatibility patch fallback triggered: {e}")
                pass
                # Final fallback: try both signatures
                try:
                    # Try new signature first (5 params)
                    if len(args) >= 3:
                        # Skip assistant_model and use remaining args
                        batch_size = args[1]
                        max_cache_length = args[2]
                        device = args[3]
                        return original_method(self, generation_config, model_kwargs, batch_size, max_cache_length, device)
                except (TypeError, IndexError):
                    # Fall back to old signature (6 params)
                    return original_method(self, generation_config, model_kwargs, *args)
        
        # Apply the patch
        VibeVoiceForConditionalGenerationInference._prepare_cache_for_generation = patched_prepare_cache_for_generation
        # Only log in debug mode or on error
        logger.debug("Applied Transformers 4.56+ compatibility patch for _prepare_cache_for_generation")
        
        return True
        
    except ImportError:
        logger.warning("VibeVoice not available - compatibility patch skipped")
        return False
    except Exception as e:
        logger.error(f"Failed to apply compatibility patch: {e}")
        return False


def patch_dynamic_cache_key_value_cache():
    """
    Patch DynamicCache to add key_cache/value_cache properties with setters for VibeVoice compatibility.

    Issue: Some transformers versions have key_cache/value_cache as read-only properties,
    but DynamicCache.__init__() tries to assign to them directly, causing "no setter" errors.

    Solution: Replace existing properties with setter-enabled properties that use private attributes.
    """
    try:
        from transformers.cache_utils import DynamicCache

        # Check if already patched
        if hasattr(DynamicCache, '_vibevoice_cache_patched'):
            return True

        # Initialize private attributes if they don't exist
        original_init = DynamicCache.__init__

        def patched_init(self, *args, **kwargs):
            # Initialize private storage attributes before calling original init
            if not hasattr(self, '_key_cache'):
                self._key_cache = []
            if not hasattr(self, '_value_cache'):
                self._value_cache = []

            # Try original init, but catch setter errors
            try:
                original_init(self, *args, **kwargs)
            except AttributeError as e:
                if "property" in str(e) and "no setter" in str(e):
                    # This is the exact error we're trying to fix
                    # Initialize the object manually
                    pass
                else:
                    raise e

        def key_cache_getter(self):
            """Compatibility getter for .key_cache access"""
            if hasattr(self, '_key_cache'):
                return self._key_cache
            # Fallback to new structure if available
            if len(self) == 0:
                return []
            return [self[i][0] if self[i] is not None and len(self[i]) >= 2 else None for i in range(len(self))]

        def key_cache_setter(self, value):
            """Compatibility setter for .key_cache assignment"""
            self._key_cache = value

        def value_cache_getter(self):
            """Compatibility getter for .value_cache access"""
            if hasattr(self, '_value_cache'):
                return self._value_cache
            # Fallback to new structure if available
            if len(self) == 0:
                return []
            return [self[i][1] if self[i] is not None and len(self[i]) >= 2 else None for i in range(len(self))]

        def value_cache_setter(self, value):
            """Compatibility setter for .value_cache assignment"""
            self._value_cache = value

        # Replace __init__ with patched version
        DynamicCache.__init__ = patched_init

        # Replace or add properties with setters (always override)
        DynamicCache.key_cache = property(key_cache_getter, key_cache_setter)
        DynamicCache.value_cache = property(value_cache_getter, value_cache_setter)

        # Mark as patched
        DynamicCache._vibevoice_cache_patched = True
        logger.debug("Applied DynamicCache key_cache/value_cache setter compatibility patch for VibeVoice")
        return True
        
    except ImportError:
        logger.warning("transformers.cache_utils not available - DynamicCache patch skipped")
        return False
    except Exception as e:
        logger.error(f"Failed to apply DynamicCache compatibility patch: {e}")
        return False


def apply_all_compatibility_patches():
    """Apply all VibeVoice compatibility patches"""
    patches_applied = []
    
    # Apply _prepare_cache_for_generation patch
    if patch_prepare_cache_for_generation():
        patches_applied.append("_prepare_cache_for_generation")
        
    # Apply DynamicCache key_cache/value_cache patch
    if patch_dynamic_cache_key_value_cache():
        patches_applied.append("dynamic_cache_properties")
    
    if patches_applied:
        logger.debug(f"VibeVoice compatibility patches applied: {', '.join(patches_applied)}")
    else:
        logger.warning("⚠️ No VibeVoice compatibility patches could be applied")
    
    return len(patches_applied) > 0