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
import sys
import types
from typing import Any

logger = logging.getLogger("VibeVoice.Compatibility")


def patch_auto_model_duplicate_registration():
    """
    TTS Audio Suite patch: make VibeVoice AutoModel registration idempotent on Transformers 5.

    VibeVoice's modular package registers several config/model pairs at import time.
    Under newer Transformers versions, repeated registration raises ValueError instead of
    silently reusing the existing mapping. We only suppress that error for VibeVoice-owned
    config classes so unrelated duplicate registrations still fail normally.
    """
    try:
        from transformers.models.auto import AutoModel, AutoModelForCausalLM

        if not hasattr(AutoModel, "_tts_suite_original_register"):
            AutoModel._tts_suite_original_register = AutoModel.register.__func__

            def patched_auto_model_register(cls, config_class, model_class, exist_ok=False):
                try:
                    return cls._tts_suite_original_register(cls, config_class, model_class, exist_ok=exist_ok)
                except ValueError as exc:
                    config_module = getattr(config_class, "__module__", "")
                    if config_module.startswith("vibevoice.") and "already used by a Transformers model" in str(exc):
                        logger.debug("Ignored duplicate AutoModel registration for %s", config_class)
                        return None
                    raise

            AutoModel.register = classmethod(patched_auto_model_register)

        if not hasattr(AutoModelForCausalLM, "_tts_suite_original_register"):
            AutoModelForCausalLM._tts_suite_original_register = AutoModelForCausalLM.register.__func__

            def patched_auto_causallm_register(cls, config_class, model_class, exist_ok=False):
                try:
                    return cls._tts_suite_original_register(cls, config_class, model_class, exist_ok=exist_ok)
                except ValueError as exc:
                    config_module = getattr(config_class, "__module__", "")
                    if config_module.startswith("vibevoice.") and "already used by a Transformers model" in str(exc):
                        logger.debug("Ignored duplicate AutoModelForCausalLM registration for %s", config_class)
                        return None
                    raise

            AutoModelForCausalLM.register = classmethod(patched_auto_causallm_register)

        return True

    except ImportError:
        logger.warning("transformers AutoModel classes not available - duplicate registration patch skipped")
        return False
    except Exception as e:
        logger.error(f"Failed to apply duplicate AutoModel registration patch: {e}")
        return False


def patch_qwen2_fast_tokenizer_import():
    """
    TTS Audio Suite patch: restore the removed Qwen2 fast-tokenizer module path.

    Some VibeVoice/Kugel code still imports
    `transformers.models.qwen2.tokenization_qwen2_fast`. On newer Transformers
    versions, the fast tokenizer lives under a different module path. We expose a
    small alias module in `sys.modules` so old imports keep working.
    """
    module_name = "transformers.models.qwen2.tokenization_qwen2_fast"

    if module_name in sys.modules:
        return True

    try:
        try:
            from transformers.models.qwen2.tokenization_qwen2 import Qwen2TokenizerFast
        except ImportError:
            from transformers.models.qwen2 import Qwen2TokenizerFast

        shim_module = types.ModuleType(module_name)
        shim_module.Qwen2TokenizerFast = Qwen2TokenizerFast
        sys.modules[module_name] = shim_module
        return True

    except ImportError:
        logger.warning("Qwen2TokenizerFast compatibility shim skipped - tokenizer class unavailable")
        return False
    except Exception as e:
        logger.error(f"Failed to apply Qwen2TokenizerFast compatibility shim: {e}")
        return False


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


def patch_vibevoice_config_num_hidden_layers():
    """
    Patch VibeVoiceConfig to add num_hidden_layers attribute.

    Issue: FushionHub/VibeVoice fork doesn't have the num_hidden_layers fix from wildminder v1.5.1.
    Transformers 4.51.3+ DynamicCache initialization requires decoder_config.num_hidden_layers.

    Fix from: https://github.com/wildminder/ComfyUI-VibeVoice/releases/tag/1.5.1
    Commit: 1ee7d7c "fix tokenizer.json issue, fix num_hidden_layers"

    Solution: Patch VibeVoiceConfig.__init__ to set num_hidden_layers from decoder_config.
    """
    try:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig

        # Check if already patched (has num_hidden_layers attribute in a fresh instance)
        if hasattr(VibeVoiceConfig, '_vibevoice_num_hidden_layers_patched'):
            return True

        # Store original __init__
        original_init = VibeVoiceConfig.__init__

        def patched_init(self, *args, **kwargs):
            """Patched __init__ that adds num_hidden_layers attribute"""
            # Call original init
            original_init(self, *args, **kwargs)

            # Add num_hidden_layers attribute from decoder_config
            # This is the exact fix from wildminder v1.5.1
            if hasattr(self, 'decoder_config') and hasattr(self.decoder_config, 'num_hidden_layers'):
                self.num_hidden_layers = self.decoder_config.num_hidden_layers
                logger.debug(f"VibeVoiceConfig: Set num_hidden_layers={self.num_hidden_layers} from decoder_config")
            else:
                # Fallback: use a reasonable default (shouldn't happen with proper models)
                logger.warning("VibeVoiceConfig: decoder_config.num_hidden_layers not found, using fallback")
                self.num_hidden_layers = 32  # Common default for 7B models

        # Apply the patch
        VibeVoiceConfig.__init__ = patched_init
        VibeVoiceConfig._vibevoice_num_hidden_layers_patched = True

        logger.debug("Applied VibeVoiceConfig.num_hidden_layers compatibility patch (wildminder v1.5.1 fix)")
        return True

    except ImportError:
        logger.warning("VibeVoice not available - num_hidden_layers patch skipped")
        return False
    except Exception as e:
        logger.error(f"Failed to apply VibeVoiceConfig num_hidden_layers patch: {e}")
        return False


def apply_all_compatibility_patches():
    """Apply all VibeVoice compatibility patches"""
    patches_applied = []

    # Apply AutoModel registration patch before importing any vibevoice.modular modules.
    if patch_auto_model_duplicate_registration():
        patches_applied.append("auto_model_duplicate_registration")

    if patch_qwen2_fast_tokenizer_import():
        patches_applied.append("qwen2_fast_tokenizer_import")

    # Apply VibeVoiceConfig num_hidden_layers patch (MUST be first - fixes the root cause)
    if patch_vibevoice_config_num_hidden_layers():
        patches_applied.append("vibevoice_config_num_hidden_layers")

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
