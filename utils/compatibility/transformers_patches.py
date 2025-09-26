"""
Transformers Compatibility Patches - DEPRECATED

⚠️ THIS MODULE IS NO LONGER USED ⚠️

Originally created to fix compatibility issues with transformers 4.46.3, but after 
upgrading to transformers 4.51.3+ (as required by VibeVoice), these patches are no 
longer needed and have been disabled.

Historical context - patches that were needed for transformers 4.46.3:
- FlashAttentionKwargs import location (moved in 4.46.3+)
- BaseStreamer import location (moved in 4.46.3+) 
- DynamicCache key_cache/value_cache properties (changed in 4.56+)
- GenerationMixin._prepare_generation_config signature (changed in 4.46.3+)
- GenerationMixin._prepare_cache_for_generation signature (changed in 4.56+)

Solution: Updated to transformers>=4.51.3 as required by VibeVoice, eliminating 
the need for these compatibility workarounds.

This file is kept for historical reference and potential future use.
"""

import warnings
from typing import Optional


class TransformersPatches:
    """Centralized transformers compatibility patches manager"""
    
    _patches_applied = set()
    
    @classmethod
    def apply_all_patches(cls, verbose: bool = True):
        """Apply all necessary transformers compatibility patches"""
        if verbose:
            print("🔧 Applying transformers compatibility patches...")

        cls.patch_flash_attention_kwargs(verbose=verbose)
        cls.patch_base_streamer(verbose=verbose)
        # Skip DynamicCache patch - users should upgrade transformers instead
        # cls.patch_dynamic_cache_properties(verbose=verbose)
        cls.patch_vibevoice_generation_methods(verbose=verbose)
        cls.patch_accelerate_compatibility(verbose=verbose)

        if verbose:
            print(f"✅ Applied {len(cls._patches_applied)} transformers compatibility patches")
    
    @classmethod
    def patch_flash_attention_kwargs(cls, verbose: bool = True):
        """
        Patch FlashAttentionKwargs import location compatibility
        
        Issue: transformers 4.46.3+ moved FlashAttentionKwargs to different module
        Affects: VibeVoice package imports
        """
        if "flash_attention_kwargs" in cls._patches_applied:
            return
            
        try:
            import transformers.modeling_flash_attention_utils
            
            # Check if FlashAttentionKwargs is missing from old location
            if not hasattr(transformers.modeling_flash_attention_utils, 'FlashAttentionKwargs'):
                # Try to import from new locations and add to old location
                try:
                    from transformers.utils import FlashAttentionKwargs
                    transformers.modeling_flash_attention_utils.FlashAttentionKwargs = FlashAttentionKwargs
                    if verbose:
                        print("   🔧 FlashAttentionKwargs patched (from transformers.utils)")
                except ImportError:
                    try:
                        from transformers.generation.utils import FlashAttentionKwargs
                        transformers.modeling_flash_attention_utils.FlashAttentionKwargs = FlashAttentionKwargs
                        if verbose:
                            print("   🔧 FlashAttentionKwargs patched (from transformers.generation.utils)")
                    except ImportError:
                        # Create dummy implementation as fallback
                        class FlashAttentionKwargs:
                            def __init__(self, **kwargs):
                                for key, value in kwargs.items():
                                    setattr(self, key, value)
                        
                        transformers.modeling_flash_attention_utils.FlashAttentionKwargs = FlashAttentionKwargs
                        if verbose:
                            print("   🔧 FlashAttentionKwargs patched (dummy implementation)")
            
            cls._patches_applied.add("flash_attention_kwargs")
            
        except Exception as e:
            warnings.warn(f"FlashAttentionKwargs patching failed: {e}")
    
    @classmethod
    def patch_base_streamer(cls, verbose: bool = True):
        """
        Patch BaseStreamer import location compatibility
        
        Issue: transformers 4.46.3+ moved BaseStreamer to different module
        Affects: VibeVoice package imports
        """
        if "base_streamer" in cls._patches_applied:
            return
            
        try:
            import transformers.generation
            
            # Check if BaseStreamer is missing from old location
            if not hasattr(transformers.generation, 'BaseStreamer'):
                try:
                    from transformers.generation.streamers import BaseStreamer
                    transformers.generation.BaseStreamer = BaseStreamer
                    if verbose:
                        print("   🔧 BaseStreamer patched (from transformers.generation.streamers)")
                except ImportError:
                    try:
                        from transformers.generation.utils import BaseStreamer
                        transformers.generation.BaseStreamer = BaseStreamer
                        if verbose:
                            print("   🔧 BaseStreamer patched (from transformers.generation.utils)")
                    except ImportError:
                        # Create dummy implementation as fallback
                        class BaseStreamer:
                            def __init__(self):
                                pass
                            
                            def put(self, value):
                                pass
                            
                            def end(self):
                                pass
                        
                        transformers.generation.BaseStreamer = BaseStreamer
                        if verbose:
                            print("   🔧 BaseStreamer patched (dummy implementation)")
            
            cls._patches_applied.add("base_streamer")
            
        except Exception as e:
            warnings.warn(f"BaseStreamer patching failed: {e}")
    
    @classmethod
    def patch_dynamic_cache_properties(cls, verbose: bool = True):
        """
        Patch DynamicCache key_cache/value_cache properties with setters

        Issue: Some transformers versions have key_cache/value_cache as read-only properties,
        but DynamicCache.__init__() tries to assign to them directly, causing "no setter" errors.
        Affects: All engines using transformers models (ChatterBox, VibeVoice, Index-TTS)
        """
        if "dynamic_cache_properties" in cls._patches_applied:
            return

        try:
            from transformers.cache_utils import DynamicCache

            # Add compatibility properties if not already patched
            if not hasattr(DynamicCache, '_tts_suite_patched'):

                # Store original __init__ method
                original_init = DynamicCache.__init__

                def patched_init(self, *args, **kwargs):
                    """Patched __init__ that handles property setter errors"""
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
                            if verbose:
                                print(f"   🔧 DynamicCache setter error caught and handled: {e}")
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

                DynamicCache._tts_suite_patched = True

                if verbose:
                    print("   🔧 DynamicCache compatibility properties with setters added")

            cls._patches_applied.add("dynamic_cache_properties")

        except Exception as e:
            warnings.warn(f"DynamicCache patching failed: {e}")
    
    @classmethod
    def patch_vibevoice_generation_methods(cls, verbose: bool = True):
        """
        Patch VibeVoice generation method signatures
        
        Issue: transformers 4.46.3+ changed method signatures for generation methods
        Affects: VibeVoice model.generate() calls
        """
        if "vibevoice_generation_methods" in cls._patches_applied:
            return
            
        try:
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            import inspect
            
            # Patch the generate method to handle signature incompatibilities
            original_generate = VibeVoiceForConditionalGenerationInference.generate
            
            def patched_generate(model_self, *args, **kwargs):
                """Patched generate with signature compatibility fixes"""
                
                # Apply method signature fixes only once per model instance
                if not hasattr(model_self, '_generation_methods_patched'):
                    
                    # Patch _prepare_cache_for_generation signature
                    original_prepare_cache = model_self._prepare_cache_for_generation
                    
                    def safe_prepare_cache_for_generation(generation_config, model_kwargs, *remaining_args):
                        try:
                            sig = inspect.signature(original_prepare_cache)
                            if len(sig.parameters) == 5:
                                # New transformers version (4.56+)
                                return original_prepare_cache(generation_config, model_kwargs, remaining_args[0], remaining_args[1], remaining_args[2])
                            else:
                                # Old transformers version (pre-4.56)
                                return original_prepare_cache(generation_config, model_kwargs, None, remaining_args[0], remaining_args[1], remaining_args[2])
                        except Exception:
                            # Fallback to try both versions
                            try:
                                return original_prepare_cache(generation_config, model_kwargs, remaining_args[0], remaining_args[1], remaining_args[2])
                            except TypeError:
                                return original_prepare_cache(generation_config, model_kwargs, None, remaining_args[0], remaining_args[1], remaining_args[2])
                    
                    model_self._prepare_cache_for_generation = safe_prepare_cache_for_generation
                    
                    # Patch _prepare_generation_config signature
                    original_prepare_gen_config = model_self._prepare_generation_config
                    
                    def safe_prepare_generation_config(*args, **kwargs):
                        try:
                            # Try calling with all arguments first
                            return original_prepare_gen_config(*args, **kwargs)
                        except TypeError as e:
                            if "takes 2 positional arguments but 3 were given" in str(e):
                                # transformers 4.46.3+ GenerationMixin._prepare_generation_config 
                                # only takes (self, generation_config), not model_kwargs
                                # But we need to ensure bos_token_id is set if missing
                                generation_config = args[0] if len(args) >= 1 else None
                                
                                if generation_config and not hasattr(generation_config, 'bos_token_id'):
                                    # Set bos_token_id from tokenizer if missing
                                    if hasattr(model_self, 'config') and hasattr(model_self.config, 'bos_token_id'):
                                        generation_config.bos_token_id = model_self.config.bos_token_id
                                    elif hasattr(model_self, 'generation_config') and hasattr(model_self.generation_config, 'bos_token_id'):
                                        generation_config.bos_token_id = model_self.generation_config.bos_token_id
                                    else:
                                        # Use a reasonable default for Qwen tokenizer
                                        generation_config.bos_token_id = 151643  # Qwen2 BOS token
                                
                                return original_prepare_gen_config(generation_config)
                            else:
                                raise e
                    
                    model_self._prepare_generation_config = safe_prepare_generation_config
                    model_self._generation_methods_patched = True
                
                return original_generate(model_self, *args, **kwargs)
            
            # Replace the generate method on the class
            VibeVoiceForConditionalGenerationInference.generate = patched_generate
            
            cls._patches_applied.add("vibevoice_generation_methods")
            
            if verbose:
                print("   🔧 VibeVoice generation methods patched")
            
        except ImportError:
            # VibeVoice not installed, skip this patch
            pass
        except Exception as e:
            warnings.warn(f"VibeVoice generation methods patching failed: {e}")

    @classmethod
    def patch_accelerate_compatibility(cls, verbose: bool = True):
        """
        Fix accelerate detection and missing function compatibility issues.

        Issue: transformers 4.56.1 + accelerate 0.20.3 compatibility problems:
        1. transformers.utils.is_accelerate_available() returns False despite accelerate being installed
        2. Missing check_tied_parameters_on_same_device function in older accelerate versions
        Affects: VibeVoice model loading with device_map
        """
        if "accelerate_compatibility" in cls._patches_applied:
            return

        try:
            # Force import accelerate first
            import accelerate

            # Force import key accelerate components that transformers needs
            from accelerate import infer_auto_device_map, dispatch_model
            from accelerate.utils import get_balanced_memory

            # Test if transformers can detect accelerate
            from transformers.utils import is_accelerate_available
            accelerate_detected = is_accelerate_available()

            if not accelerate_detected:
                if verbose:
                    print("   🔧 Fixing broken transformers accelerate detection")

                # Comprehensive monkey patch for all accelerate detection functions
                import transformers.utils
                def fixed_is_accelerate_available():
                    try:
                        import accelerate
                        return True
                    except ImportError:
                        return False

                # Replace the broken detection function in all locations
                transformers.utils.is_accelerate_available = fixed_is_accelerate_available

                try:
                    import transformers.modeling_utils
                    transformers.modeling_utils.is_accelerate_available = fixed_is_accelerate_available
                except:
                    pass

                try:
                    import transformers.utils.import_utils
                    transformers.utils.import_utils.is_accelerate_available = fixed_is_accelerate_available
                except:
                    pass

                try:
                    import transformers
                    if hasattr(transformers, 'is_accelerate_available'):
                        transformers.is_accelerate_available = fixed_is_accelerate_available
                except:
                    pass

            # Add missing functions that newer transformers expects
            def check_tied_parameters_on_same_device(model, device_map=None):
                """
                Dummy implementation of missing accelerate function.
                This function checks if tied parameters are on the same device.

                Args:
                    model: The model to check
                    device_map: Optional device mapping

                Returns:
                    bool: True if tied parameters are on same device
                """
                return True  # For single GPU setups, this is always true

            def dispatch_model(model, device_map=None, main_device=None, state_dict=None,
                             offload_dir=None, offload_index=None, **kwargs):
                """
                Dummy implementation of missing accelerate.dispatch_model function.
                For single GPU setups, we just move the model to the target device.

                Args:
                    model: The model to dispatch
                    device_map: Device mapping (ignored in dummy implementation)
                    main_device: Main device to use
                    **kwargs: Additional arguments (ignored)

                Returns:
                    The model (moved to device if specified)
                """
                if main_device is not None:
                    model = model.to(main_device)
                elif device_map == "cuda" or (isinstance(device_map, dict) and "cuda" in str(device_map)):
                    model = model.cuda()
                return model

            # Only add missing functions if we have compatibility issues
            needs_function_patches = False

            # Check if critical functions are missing (indicating compatibility issues)
            try:
                import accelerate.utils
                if not hasattr(accelerate.utils, 'check_tied_parameters_on_same_device'):
                    needs_function_patches = True
            except:
                pass

            try:
                import accelerate
                if not hasattr(accelerate, 'dispatch_model'):
                    needs_function_patches = True
            except:
                pass

            # Only apply function patches if we detected issues OR accelerate detection failed
            if not accelerate_detected or needs_function_patches:

                if verbose and needs_function_patches:
                    print("   🔧 Adding missing accelerate functions")

                # Add missing functions to accelerate.utils
                try:
                    import accelerate.utils
                    if not hasattr(accelerate.utils, 'check_tied_parameters_on_same_device'):
                        accelerate.utils.check_tied_parameters_on_same_device = check_tied_parameters_on_same_device
                except Exception as e:
                    if verbose and not accelerate_detected:
                        print(f"   ⚠️ Could not add to accelerate.utils: {e}")

                # Add dispatch_model to accelerate module
                try:
                    import accelerate
                    if not hasattr(accelerate, 'dispatch_model'):
                        accelerate.dispatch_model = dispatch_model
                except Exception as e:
                    if verbose and not accelerate_detected:
                        print(f"   ⚠️ Could not add dispatch_model to accelerate: {e}")

                # Also add to transformers modules that might import it
                try:
                    import transformers.modeling_utils
                    if not hasattr(transformers.modeling_utils, 'check_tied_parameters_on_same_device'):
                        transformers.modeling_utils.check_tied_parameters_on_same_device = check_tied_parameters_on_same_device
                    if not hasattr(transformers.modeling_utils, 'dispatch_model'):
                        transformers.modeling_utils.dispatch_model = dispatch_model
                except Exception as e:
                    if verbose and not accelerate_detected:
                        print(f"   ⚠️ Could not add to transformers.modeling_utils: {e}")

                # Add to global namespace as well (for direct imports)
                try:
                    import builtins
                    builtins.check_tied_parameters_on_same_device = check_tied_parameters_on_same_device
                    builtins.dispatch_model = dispatch_model
                except Exception as e:
                    if verbose and not accelerate_detected:
                        print(f"   ⚠️ Could not add to global namespace: {e}")

                # Add to any vibevoice modules that might need it
                try:
                    import sys
                    for module_name, module in sys.modules.items():
                        if 'vibevoice' in module_name.lower() and hasattr(module, '__dict__'):
                            if 'check_tied_parameters_on_same_device' not in module.__dict__:
                                module.__dict__['check_tied_parameters_on_same_device'] = check_tied_parameters_on_same_device
                            if 'dispatch_model' not in module.__dict__:
                                module.__dict__['dispatch_model'] = dispatch_model
                except Exception as e:
                    if verbose and not accelerate_detected:
                        print(f"   ⚠️ Could not add to vibevoice modules: {e}")

            cls._patches_applied.add("accelerate_compatibility")

            # Only log if patches were actually needed
            if verbose and not accelerate_detected:
                print("   ✅ Accelerate compatibility patches applied")
            elif verbose:
                # Silent for working installations - no log pollution
                pass

        except ImportError as e:
            if verbose:
                print(f"   ❌ accelerate not available: {e}")
        except Exception as e:
            warnings.warn(f"Accelerate compatibility patching failed: {e}")

    @classmethod
    def get_applied_patches(cls):
        """Get list of applied patches"""
        return list(cls._patches_applied)
    
    @classmethod
    def is_patch_applied(cls, patch_name: str) -> bool:
        """Check if a specific patch has been applied"""
        return patch_name in cls._patches_applied


# Convenience function for easy import
def apply_transformers_patches(verbose: bool = True):
    """Apply all transformers compatibility patches"""
    TransformersPatches.apply_all_patches(verbose=verbose)


# Auto-apply on import for critical patches - DISABLED
# These patches are no longer needed with transformers 4.51.3+ and cause compatibility issues
if False:  # Completely disabled - was causing DynamicCache errors with newer transformers
    # Only apply critical patches on import to avoid side effects
    TransformersPatches.patch_flash_attention_kwargs(verbose=False)
    TransformersPatches.patch_base_streamer(verbose=False)