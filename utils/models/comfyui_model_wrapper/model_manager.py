"""
ComfyUI TTS Model Manager
"""

import weakref
import gc
from typing import Optional, Any, Dict

from .base_wrapper import ComfyUIModelWrapper, ModelInfo, COMFYUI_AVAILABLE, model_management


class ComfyUITTSModelManager:
    """
    Manager that integrates TTS models with ComfyUI's model management system.
    
    This replaces static caches with ComfyUI-managed model loading/unloading.
    """
    
    def __init__(self):
        self._model_cache: Dict[str, ComfyUIModelWrapper] = {}

    def _offload_conflicting_tts_models(self, engine: str, target_device: str, keep_model_key: Optional[str] = None) -> int:
        """
        Move conflicting GPU-resident TTS models to CPU before loading/moving another one.

        This keeps fresh load_model() behavior aligned with ensure_device(): if a new
        TTS model wants CUDA, other cached TTS engines should not stay in VRAM.
        """
        if not str(target_device).startswith('cuda'):
            return 0

        models_to_offload = []
        for cache_key, wrapper in list(self._model_cache.items()):
            should_offload = (
                wrapper.model_info.model_type == "tts" and
                wrapper._is_loaded_on_gpu and
                (wrapper.model_info.engine != engine or
                 (keep_model_key is not None and cache_key != keep_model_key))
            )
            if should_offload:
                models_to_offload.append((cache_key, wrapper))

        if not models_to_offload:
            return 0

        print(f"🗑️ Moving {len(models_to_offload)} TTS model(s) to CPU to make room for {engine}")
        for cache_key, wrapper in models_to_offload:
            try:
                wrapper.partially_unload('cpu', wrapper._memory_size)
            except Exception as e:
                print(f"⚠️ Failed to move {wrapper.model_info.engine} to CPU, removing instead: {e}")
                self.remove_model(cache_key)

        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            import time
            time.sleep(0.1)

        return len(models_to_offload)
        
    def load_model(self, 
                   model_factory_func, 
                   model_key: str,
                   model_type: str,
                   engine: str, 
                   device: str,
                   force_reload: bool = False,
                   **factory_kwargs) -> ComfyUIModelWrapper:
        """
        Load a model using ComfyUI's model management system.
        
        Args:
            model_factory_func: Function that creates the model
            model_key: Unique key for caching
            model_type: Type of model ("tts", "vc", etc.)  
            engine: Engine name ("chatterbox", "f5tts", etc.)
            device: Target device
            **factory_kwargs: Arguments for model factory function
            
        Returns:
            ComfyUI-wrapped model
        """
        # No more shadow storage - if model was destroyed, create completely fresh
        # Check if already cached
        if model_key in self._model_cache and not force_reload:
            wrapper = self._model_cache[model_key]
            is_valid = getattr(wrapper, '_is_valid_for_reuse', True)
            print(f"🔍 Cache check for {model_type} ({engine}): valid={is_valid}, force_reload={force_reload}")
            
            # Check if cached model is still valid for reuse
            if not is_valid:
                # For Higgs Audio with CUDA graph corruption, try to reinitialize in-place
                if engine == "higgs_audio":
                    print(f"🔄 Attempting in-place reinitializion of corrupted {engine} model to avoid memory conflicts")
                    try:
                        # Reset CUDA graph state without creating new model
                        if hasattr(wrapper.model, 'engine') and hasattr(wrapper.model.engine, 'cuda_graphs_initialized'):
                            wrapper.model.engine.cuda_graphs_initialized = False
                            print(f"✅ Reset CUDA graph state for existing model")
                        
                        # Move back to GPU for reinit
                        wrapper.model_load(device)
                        # Mark as valid again
                        wrapper._is_valid_for_reuse = True
                        print(f"✅ Successfully reinitialized {engine} model in-place")
                        return wrapper
                    except Exception as e:
                        print(f"⚠️ In-place reinit failed: {e}, falling back to full recreation")
                
                # For VibeVoice, try to reinitialize corrupted model state
                # Unlike Higgs Audio, VibeVoice doesn't use CUDA graphs so should be recoverable
                elif engine == "vibevoice":
                    print(f"🔄 VibeVoice: Attempting to recover from CPU offloading corruption")
                    try:
                        # Clear any cached internal state that might be corrupted
                        if hasattr(wrapper.model, '_past_key_values'):
                            wrapper.model._past_key_values = None
                        if hasattr(wrapper.model, '_cache'):
                            wrapper.model._cache = None

                        # Reset model to evaluation mode and clear gradients
                        wrapper.model.eval()
                        if hasattr(wrapper.model, 'zero_grad'):
                            wrapper.model.zero_grad()

                        # Move back to GPU with proper state reset
                        wrapper.model_load(device)
                        # Mark as valid again
                        wrapper._is_valid_for_reuse = True
                        print(f"✅ Successfully recovered VibeVoice model from corruption")
                        return wrapper
                    except Exception as e:
                        print(f"⚠️ VibeVoice recovery failed: {e}, falling back to full recreation")

                # For IndexTTS-2, try to recover from device mismatch after CPU offloading
                elif engine == "index_tts":
                    print(f"🔄 IndexTTS-2: Attempting to recover from CPU offloading device mismatch")
                    try:
                        # IndexTTS-2 has multiple model components that need device synchronization
                        # Clear any device-cached state
                        if hasattr(wrapper.model, '_model_config'):
                            wrapper.model._model_config = None

                        # Reset model to evaluation mode
                        if hasattr(wrapper.model, 'eval'):
                            wrapper.model.eval()

                        # Force device reload for all model components
                        wrapper.model_load(device)

                        # Mark as valid again
                        wrapper._is_valid_for_reuse = True
                        print(f"✅ Successfully recovered IndexTTS-2 model from device mismatch")
                        return wrapper
                    except Exception as e:
                        print(f"⚠️ IndexTTS-2 recovery failed: {e}, falling back to full recreation")
                
                print(f"🗑️ Removing invalid cached model: {model_type} ({engine}) - corrupted by previous unload")
                self.remove_model(model_key)
                # Continue to create new model below
            else:
                print(f"♻️ Reusing valid cached model: {model_type} ({engine})")

                # Before loading to GPU, clear other models to make room
                if wrapper.current_device != device and device != 'auto' and device.startswith('cuda'):
                    # Clear models from different engines to free VRAM
                    cached_models = list(self._model_cache.keys())
                    models_to_clear = []

                    for cache_key in cached_models:
                        cached_wrapper = self._model_cache[cache_key]
                        # Clear TTS models from different engines that are on GPU
                        if (cached_wrapper.model_info.engine != engine and
                            cached_wrapper.model_info.model_type == "tts" and
                            cached_wrapper._is_loaded_on_gpu):
                            models_to_clear.append(cache_key)

                    if models_to_clear:
                        print(f"🗑️ Clearing {len(models_to_clear)} TTS models to free VRAM for reused model")
                        for key in models_to_clear:
                            self.remove_model(key)

                # Ensure model is loaded on correct device
                if wrapper.current_device != device and device != 'auto':
                    wrapper.model_load(device)
                return wrapper
        elif force_reload and model_key in self._model_cache:
            wrapper = self._model_cache[model_key]
            
            # For Higgs Audio, try in-place reinitialization instead of full recreation
            if engine == "higgs_audio":
                print(f"🔄 Force reload: attempting in-place reinitializion of {engine} model to avoid memory conflicts")
                try:
                    # Reset CUDA graph state without creating new model
                    if hasattr(wrapper.model, 'engine') and hasattr(wrapper.model.engine, 'cuda_graphs_initialized'):
                        wrapper.model.engine.cuda_graphs_initialized = False
                        print(f"✅ Reset CUDA graph state for existing model")

                    # Move back to GPU for reinit
                    wrapper.model_load(device)
                    # Mark as valid again
                    wrapper._is_valid_for_reuse = True
                    print(f"✅ Successfully reinitialized {engine} model in-place (force reload)")
                    return wrapper
                except Exception as e:
                    print(f"⚠️ Force reload in-place reinit failed: {e}, falling back to full recreation")

            # For IndexTTS-2, try in-place device synchronization on force reload
            elif engine == "index_tts":
                print(f"🔄 Force reload: attempting IndexTTS-2 device synchronization")
                try:
                    # Clear device-cached state
                    if hasattr(wrapper.model, '_model_config'):
                        wrapper.model._model_config = None

                    # Force device reload for all model components
                    wrapper.model_load(device)
                    # Mark as valid again
                    wrapper._is_valid_for_reuse = True
                    print(f"✅ Successfully reloaded IndexTTS-2 model with device sync (force reload)")
                    return wrapper
                except Exception as e:
                    print(f"⚠️ IndexTTS-2 force reload failed: {e}, falling back to full recreation")
            
            print(f"🔄 Force reloading {model_type} ({engine}) - removing from cache")
            self.remove_model(model_key)

        # CRITICAL: Unload old same-engine model variants BEFORE ComfyUI's free_memory
        # (prevents loading multiple Qwen3-TTS variants with different attn_implementation, etc.)
        if model_type == "tts" and engine != "":
            cached_models = list(self._model_cache.keys())
            for existing_key in cached_models:
                if existing_key.startswith(f"{engine}_tts_") and existing_key != model_key:
                    wrapper = self._model_cache.get(existing_key)
                    if wrapper and wrapper._is_loaded_on_gpu:
                        print(f"🗑️ Unloading old {engine} model variant to prevent VRAM accumulation")
                        self.remove_model(existing_key)
                        break  # Only unload one old variant

        # Aggressive memory management before loading new model
        memory_freed = 0  # Track if ComfyUI freed memory (indicates VRAM pressure)
        if COMFYUI_AVAILABLE and model_management is not None and device != 'cpu':
            try:
                # Free up memory aggressively - request 3GB to ensure space for new model
                if hasattr(model_management, 'free_memory') and callable(getattr(model_management, 'free_memory', None)):
                    if hasattr(model_management, 'get_torch_device'):
                        torch_device = model_management.get_torch_device()
                        # Request 3GB of free VRAM (aggressive cleanup for TTS models)
                        memory_freed = model_management.free_memory(3 * 1024 * 1024 * 1024, torch_device)
                        if memory_freed and memory_freed > 0:
                            print(f"🧹 Freed {memory_freed // 1024 // 1024}MB VRAM for new {model_type} model")

            except Exception as e:
                # Silently ignore memory management errors to avoid spam
                pass

        if model_type == "tts" and engine != "" and str(device).startswith('cuda'):
            self._offload_conflicting_tts_models(engine, device, keep_model_key=model_key)
        
        # Create the model
        # print(f"🔧 Creating new {model_type} model ({engine}) on {device} - fresh instance after cache invalidation")
        
        # Higgs Audio now uses deferred CUDA graph initialization to prevent corruption
        if device.startswith('cuda') and engine == "higgs_audio":
            # print(f"📝 Creating fresh {engine} model (CUDA graphs deferred until first inference)")
            try:
                import gc
                gc.collect()
            except Exception as gc_error:
                print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")
        
        # Call factory function with config if provided, otherwise use **kwargs
        if 'config' in factory_kwargs:
            # New pattern: factory receives ModelLoadConfig object
            config = factory_kwargs.pop('config')
            model = model_factory_func(config)
        else:
            # Legacy pattern: factory receives **kwargs
            factory_kwargs['device'] = device
            model = model_factory_func(**factory_kwargs)
        
        # Calculate memory usage
        memory_size = ComfyUIModelWrapper.calculate_model_memory(model)
        
        # Create model info - for stateless wrappers, use a generic engine name to prevent CUDA graph handling
        actual_engine = engine
        original_engine = None
        if hasattr(model, '_wrapped_engine') and engine == "higgs_audio":
            # This is a stateless wrapper - use generic name to prevent ComfyUI from doing special CUDA handling
            original_engine = engine  # Store original engine name
            actual_engine = "stateless_tts"

        model_info = ModelInfo(
            model=model,
            model_type=model_type,
            engine=actual_engine,  # Use generic name for stateless wrappers
            device=device,
            memory_size=memory_size,
            load_device=device,
            original_engine=original_engine  # Store original engine before masking
        )
        
        # Wrap for ComfyUI, passing cache_key so wrapper knows which specific model it is
        wrapper = ComfyUIModelWrapper(model, model_info, cache_key=model_key)

        # Cache the wrapper
        self._model_cache[model_key] = wrapper
        
        # Register with ComfyUI using the proper load_models_gpu method
        if COMFYUI_AVAILABLE and model_management is not None:
            # Try the safer manual approach first since load_models_gpu seems to have issues
            try:
                if hasattr(model_management, 'LoadedModel') and hasattr(model_management, 'current_loaded_models'):
                    loaded_model = model_management.LoadedModel(wrapper)
                    if model is not None:
                        loaded_model.real_model = weakref.ref(model)
                        # Set up the finalizer that ComfyUI expects
                        if hasattr(model_management, 'cleanup_models'):
                            loaded_model.model_finalizer = weakref.finalize(model, model_management.cleanup_models)
                        else:
                            # Create a dummy finalizer that doesn't crash
                            loaded_model.model_finalizer = weakref.finalize(model, lambda: None)
                    else:
                        loaded_model.real_model = weakref.ref(wrapper)
                        loaded_model.model_finalizer = weakref.finalize(wrapper, lambda: None)
                    
                    # Keep a strong reference to our wrapper to prevent garbage collection
                    # This ensures LoadedModel.model property doesn't return None
                    loaded_model._tts_wrapper_ref = wrapper
                    
                    model_management.current_loaded_models.insert(0, loaded_model)  # Insert at 0 like ComfyUI does
                    total_models = len(model_management.current_loaded_models)
                    # print(f"📦 {model_type.title()} ready with ComfyUI integration (#{total_models})")
                else:
                    print(f"⚠️ ComfyUI LoadedModel or current_loaded_models not available")
            except Exception as e:
                print(f"⚠️ Failed to register with ComfyUI model management: {e}")
                
        return wrapper
    
    def get_model(self, model_key: str) -> Optional[ComfyUIModelWrapper]:
        """Get a cached model by key"""
        return self._model_cache.get(model_key)
        
    def remove_model(self, model_key: str) -> bool:
        """Remove a model from cache and ComfyUI tracking"""
        if model_key in self._model_cache:
            wrapper = self._model_cache[model_key]

            # With stateless wrapper, Higgs Audio models can now be safely unloaded
            print(f"🗑️ Attempting to unload {wrapper.model_info.engine} model (stateless wrapper enabled)")

            # Measure VRAM before unload
            vram_before = 0
            if wrapper._is_loaded_on_gpu:
                import torch
                if torch.cuda.is_available():
                    vram_before = torch.cuda.memory_allocated()

            # Normal destruction for all engines
            self._model_cache.pop(model_key)

            # Remove from ComfyUI tracking if available
            if COMFYUI_AVAILABLE and model_management is not None:
                try:
                    if hasattr(model_management, 'current_loaded_models'):
                        tracked_models = model_management.current_loaded_models
                        matches = [
                            loaded_model for loaded_model in list(tracked_models)
                            if loaded_model is wrapper
                            or getattr(loaded_model, '_tts_wrapper_ref', None) is wrapper
                        ]
                        for loaded_model in matches:
                            tracked_models.remove(loaded_model)
                        if matches:
                            print(f"🗑️ Removed model from ComfyUI tracking")
                except Exception as e:
                    print(f"⚠️ Failed to remove from ComfyUI tracking: {e}")

            # Permanent removal should release engines that provide a direct
            # destruction path. Normal ComfyUI offloading still goes through
            # wrapper.model_unload()/partially_unload() and remains reusable.
            from .engine_handlers import get_engine_handler
            handler = get_engine_handler(wrapper.model_info.engine)
            release = getattr(handler, 'release', None)
            if callable(release):
                release(wrapper)
            else:
                wrapper.model_unload()

            # Explicitly delete wrapper to release all references
            del wrapper

            # Force garbage collection
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Measure VRAM after unload and report actual freed amount
            if vram_before > 0:
                if torch.cuda.is_available():
                    vram_after = torch.cuda.memory_allocated()
                    freed = vram_before - vram_after
                    print(f"📊 VRAM: {vram_before // 1024 // 1024}MB → {vram_after // 1024 // 1024}MB (freed {freed // 1024 // 1024}MB)")

            return True
        return False
        
    def clear_cache(self, model_type: Optional[str] = None, engine: Optional[str] = None):
        """Clear cached models with optional filtering"""
        keys_to_remove = []
        
        for key, wrapper in self._model_cache.items():
            should_remove = True
            
            if model_type and wrapper.model_info.model_type != model_type:
                should_remove = False
            if engine and wrapper.model_info.engine != engine:  
                should_remove = False
                
            if should_remove:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            self.remove_model(key)
            
        print(f"🧹 Cleared {len(keys_to_remove)} models from cache")
        
    def ensure_device(self, engine_name: str, target_device: str, model_cache_key: str = None) -> bool:
        """
        Ensure a model is on the target device, clearing other models if needed.

        This is the SINGLE point where device movement happens - all engines should call this
        instead of doing their own .to(device) calls.

        Args:
            engine_name: Name of the engine requesting device movement
            target_device: Target device ("cuda", "cpu", etc.)
            model_cache_key: Optional specific model cache key (to distinguish vibevoice-7B from vibevoice-1.5B, etc.)

        Returns:
            True if successful, False otherwise
        """
        # Before moving to GPU, ALWAYS move other TTS models to CPU to free VRAM
        # (Don't delete them - keep in cache for fast reload)
        if target_device.startswith('cuda'):
            cleared = self._offload_conflicting_tts_models(
                engine_name,
                target_device,
                keep_model_key=model_cache_key,
            )
            if cleared:
                print(f"✅ CLEARING COMPLETE: VRAM freed and ready for {engine_name}")

        # Find the model for this engine
        wrapper = None
        if model_cache_key:
            # If specific cache key provided, use it directly
            wrapper = self._model_cache.get(model_cache_key)
            if not wrapper:
                print(f"⚠️ ensure_device: Specified model cache key not found: {model_cache_key[:60]}...")
                return False
        else:
            # Fallback: find first model with matching engine (old behavior)
            for cache_key, cached_wrapper in self._model_cache.items():
                if cached_wrapper.model_info.engine == engine_name:
                    wrapper = cached_wrapper
                    break

            if not wrapper:
                # Model not in unified cache - engine must be managing it itself
                # We've already cleared other models, so return False to let engine handle movement
                return False

        # Check if already on target device
        if wrapper.current_device == target_device:
            return True

        # Move the model to target device
        try:
            # Pass flag to prevent recursion (model_load shouldn't call ensure_device again)
            wrapper.model_load(target_device, _from_ensure_device=True)
            return True
        except Exception as e:
            print(f"❌ Failed to move {engine_name} to {target_device}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_memory = sum(w.loaded_size() for w in self._model_cache.values())
        by_type = {}
        by_engine = {}

        for wrapper in self._model_cache.values():
            model_type = wrapper.model_info.model_type
            engine = wrapper.model_info.engine

            by_type[model_type] = by_type.get(model_type, 0) + 1
            by_engine[engine] = by_engine.get(engine, 0) + 1

        return {
            'total_models': len(self._model_cache),
            'total_memory_mb': total_memory // 1024 // 1024,
            'by_type': by_type,
            'by_engine': by_engine,
            'comfyui_integration': COMFYUI_AVAILABLE
        }


# Global instance for all TTS models
tts_model_manager = ComfyUITTSModelManager()
