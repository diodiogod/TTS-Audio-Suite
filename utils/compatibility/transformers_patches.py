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
        # Skip DynamicCache patch - causes property setter conflicts with transformers 4.46.3+
        # cls.patch_dynamic_cache_properties(verbose=verbose)  
        cls.patch_vibevoice_generation_methods(verbose=verbose)
        
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
        Patch DynamicCache key_cache/value_cache properties
        
        Issue: transformers 4.56+ changed DynamicCache internal structure
        Affects: VibeVoice cache access patterns
        """
        if "dynamic_cache_properties" in cls._patches_applied:
            return
            
        try:
            from transformers.cache_utils import DynamicCache
            
            # Add compatibility properties if not already patched
            if not hasattr(DynamicCache, '_tts_suite_patched'):
                
                def key_cache_property(self):
                    """Compatibility property for .key_cache access"""
                    if len(self) == 0:
                        return []
                    return [self[i][0] if self[i] is not None and len(self[i]) >= 2 else None for i in range(len(self))]
                
                def value_cache_property(self):
                    """Compatibility property for .value_cache access"""  
                    if len(self) == 0:
                        return []
                    return [self[i][1] if self[i] is not None and len(self[i]) >= 2 else None for i in range(len(self))]
                
                # Add properties to the class
                DynamicCache.key_cache = property(key_cache_property)
                DynamicCache.value_cache = property(value_cache_property)
                DynamicCache._tts_suite_patched = True
                
                if verbose:
                    print("   🔧 DynamicCache compatibility properties added")
            
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


# Auto-apply on import for critical patches
if __name__ != "__main__":
    # Only apply critical patches on import to avoid side effects
    TransformersPatches.patch_flash_attention_kwargs(verbose=False)
    TransformersPatches.patch_base_streamer(verbose=False)