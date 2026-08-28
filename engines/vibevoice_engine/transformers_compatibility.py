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
import copy
import sys
import types
from typing import Any

from packaging import version

logger = logging.getLogger("VibeVoice.Compatibility")


def _is_transformers_5_or_newer() -> bool:
    try:
        import transformers
        return version.parse(transformers.__version__) >= version.parse("5.0.0")
    except Exception:
        return False


def install_transformers5_import_shims():
    """Install the import compatibility needed by legacy VibeVoice."""
    if not _is_transformers_5_or_newer():
        return False

    # TTS Audio Suite patch: Transformers 5 folded the fast Qwen2 tokenizer
    # into Qwen2Tokenizer, while the legacy VibeVoice package imports the old
    # module path directly.
    fast_module_name = "transformers.models.qwen2.tokenization_qwen2_fast"
    if fast_module_name not in sys.modules:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        fast_module = types.ModuleType(fast_module_name)
        fast_module.Qwen2TokenizerFast = Qwen2Tokenizer
        sys.modules[fast_module_name] = fast_module

    return True


def import_vibevoice_with_transformers5_compatibility():
    """Import VibeVoice while allowing its legacy auto-class registrations."""
    install_transformers5_import_shims()

    from transformers import AutoModel, AutoModelForCausalLM

    registrations = (AutoModel, AutoModelForCausalLM)
    originals = {auto_class: auto_class.register for auto_class in registrations}
    original_descriptors = {
        auto_class: auto_class.__dict__.get("register") for auto_class in registrations
    }

    if _is_transformers_5_or_newer():
        for auto_class in registrations:
            original_register = originals[auto_class]

            def compatible_register(config_class, model_class, exist_ok=False, _original=original_register):
                is_vibevoice = getattr(config_class, "__module__", "").startswith("vibevoice.")
                return _original(config_class, model_class, exist_ok=exist_ok or is_vibevoice)

            auto_class.register = compatible_register

    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    finally:
        for auto_class, original_descriptor in original_descriptors.items():
            if original_descriptor is None:
                delattr(auto_class, "register")
            else:
                auto_class.register = original_descriptor

    return VibeVoiceForConditionalGenerationInference, VibeVoiceProcessor


def patch_transformers5_generation_api():
    """Adapt legacy VibeVoice generation calls to Transformers 5 APIs."""
    if not _is_transformers_5_or_newer():
        return False

    try:
        VibeVoiceForConditionalGenerationInference, _ = (
            import_vibevoice_with_transformers5_compatibility()
        )
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
        from transformers.generation import GenerationMixin

        # TTS Audio Suite patch: the legacy model reads config.torch_dtype.
        # Route that old attribute name to Transformers 5's dtype field without
        # triggering the framework's deprecation warning.
        VibeVoiceConfig.attribute_map = {
            **getattr(VibeVoiceConfig, "attribute_map", {}),
            "torch_dtype": "dtype",
        }

        if not hasattr(DPMSolverMultistepScheduler, "_tts_cpu_init_patched"):
            original_scheduler_init = DPMSolverMultistepScheduler.__init__

            def compatible_scheduler_init(self, *args, **kwargs):
                # TTS Audio Suite patch: Transformers initializes model modules
                # under a meta-device context. Scheduler tensors are runtime
                # state, not model parameters, and must be materialized on CPU.
                import torch

                with torch.device("cpu"):
                    original_scheduler_init(self, *args, **kwargs)

            DPMSolverMultistepScheduler.__init__ = compatible_scheduler_init
            DPMSolverMultistepScheduler._tts_cpu_init_patched = True

        if not hasattr(VibeVoiceForConditionalGenerationInference, "_tts_t5_generation_patched"):
            original_prepare_generation_config = GenerationMixin._prepare_generation_config
            original_prepare_cache = GenerationMixin._prepare_cache_for_generation
            original_prepare_inputs = VibeVoiceForConditionalGenerationInference.prepare_inputs_for_generation

            def compatible_prepare_generation_config(self, generation_config, *args, **kwargs):
                # TTS Audio Suite patch: legacy VibeVoice passes the removed
                # `use_model_defaults` positional boolean.
                if args and isinstance(args[0], bool):
                    args = args[1:]
                if args:
                    raise TypeError("Unexpected positional generation arguments")

                # Transformers 5 deprecates mixing an explicit GenerationConfig
                # with generation kwargs. Merge recognized/custom generation
                # settings into a copy and pass only real model inputs onward.
                if generation_config is not None:
                    generation_config = copy.deepcopy(generation_config)
                    for custom_key in (
                        "speech_start_id",
                        "speech_end_id",
                        "speech_diffusion_id",
                    ):
                        if custom_key in kwargs:
                            setattr(generation_config, custom_key, kwargs.pop(custom_key))
                    kwargs = generation_config.update(**kwargs)
                return original_prepare_generation_config(self, generation_config, **kwargs)

            def compatible_prepare_cache(
                self,
                generation_config,
                model_kwargs,
                generation_mode,
                batch_size,
                max_cache_length,
                *args,
            ):
                # TTS Audio Suite patch: Transformers 5 removed the trailing
                # device argument from this hook.
                return original_prepare_cache(
                    self,
                    generation_config,
                    model_kwargs,
                    generation_mode,
                    batch_size,
                    max_cache_length,
                )

            def compatible_prepare_inputs(self, *args, **kwargs):
                result = original_prepare_inputs(self, *args, **kwargs)
                if isinstance(result, dict):
                    result.setdefault("inputs_embeds", None)
                return result

            VibeVoiceForConditionalGenerationInference._prepare_generation_config = compatible_prepare_generation_config
            VibeVoiceForConditionalGenerationInference._prepare_cache_for_generation = compatible_prepare_cache
            VibeVoiceForConditionalGenerationInference.prepare_inputs_for_generation = compatible_prepare_inputs

            # TTS Audio Suite patch: the official 1.5B checkpoint intentionally
            # omits lm_head.weight because it is tied to decoder embeddings.
            # Transformers reports the key before tie_weights() runs unless the
            # model declares it as an expected omission.
            ignored_missing = list(
                getattr(
                    VibeVoiceForConditionalGenerationInference,
                    "_keys_to_ignore_on_load_missing",
                    None,
                )
                or []
            )
            if r"lm_head\.weight" not in ignored_missing:
                ignored_missing.append(r"lm_head\.weight")
            VibeVoiceForConditionalGenerationInference._keys_to_ignore_on_load_missing = ignored_missing

            for model_class in (
                VibeVoiceForConditionalGeneration,
                VibeVoiceForConditionalGenerationInference,
            ):
                original_tie_weights = model_class.tie_weights

                if model_class is VibeVoiceForConditionalGenerationInference:
                    def compatible_tie_weights(self, *args, **kwargs):
                        # TTS Audio Suite patch: VibeVoiceConfig keeps this
                        # flag on decoder_config. Reading the top-level config
                        # leaves the 1.5B lm_head randomly initialized.
                        decoder_config = getattr(self.config, "decoder_config", None)
                        if not getattr(decoder_config, "tie_word_embeddings", False):
                            return
                        if hasattr(self, "lm_head") and hasattr(
                            self.model.language_model, "embed_tokens"
                        ):
                            self.lm_head.weight = self.model.language_model.embed_tokens.weight
                else:
                    def compatible_tie_weights(self, *args, _original=original_tie_weights, **kwargs):
                        return _original(self)

                model_class.tie_weights = compatible_tie_weights

            VibeVoiceForConditionalGenerationInference._tts_t5_generation_patched = True

        return True
    except Exception as e:
        logger.error(f"Failed to apply Transformers 5 VibeVoice generation compatibility: {e}")
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

        transformers5 = _is_transformers_5_or_newer()

        # Initialize private attributes if they don't exist
        original_init = DynamicCache.__init__

        def patched_init(self, *args, **kwargs):
            if transformers5:
                return original_init(self, *args, **kwargs)

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
            if transformers5 and hasattr(self, "layers"):
                return [getattr(layer, "keys", None) for layer in self.layers]
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
            if transformers5 and hasattr(self, "layers"):
                return [getattr(layer, "values", None) for layer in self.layers]
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
    transformers5 = _is_transformers_5_or_newer()

    if install_transformers5_import_shims():
        patches_applied.append("transformers5_import_shims")

    # Apply VibeVoiceConfig num_hidden_layers patch (MUST be first - fixes the root cause)
    if patch_vibevoice_config_num_hidden_layers():
        patches_applied.append("vibevoice_config_num_hidden_layers")

    # This adapter targets the Transformers 4.56 signature. Transformers 5 is
    # handled by patch_transformers5_generation_api() below.
    if not transformers5 and patch_prepare_cache_for_generation():
        patches_applied.append("_prepare_cache_for_generation")

    # Apply DynamicCache key_cache/value_cache patch
    if patch_dynamic_cache_key_value_cache():
        patches_applied.append("dynamic_cache_properties")

    # Apply this last because it replaces the older Transformers 4.56 cache
    # adapter with the Transformers 5 signatures when necessary.
    if patch_transformers5_generation_api():
        patches_applied.append("transformers5_generation_api")

    if patches_applied:
        logger.debug(f"VibeVoice compatibility patches applied: {', '.join(patches_applied)}")
    else:
        logger.warning("⚠️ No VibeVoice compatibility patches could be applied")

    return len(patches_applied) > 0
