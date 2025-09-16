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
            
        # Aggressive memory management before loading new model
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
                
                # Also try manual cleanup of our own TTS model cache
                # Clear models from other engines to make room
                if model_type == "tts" and engine != "":
                    # Get current cache stats
                    cached_models = list(self._model_cache.keys())
                    models_to_clear = []
                    
                    for cache_key in cached_models:
                        wrapper = self._model_cache[cache_key]
                        should_clear = False
                        
                        # Clear models from different engines to free VRAM
                        if wrapper.model_info.engine != engine and wrapper.model_info.model_type == "tts":
                            should_clear = True
                        
                        # VibeVoice-specific: Use CPU migration instead of direct clearing
                        # This allows model reuse while preventing RAM accumulation
                        elif engine == "vibevoice" and wrapper.model_info.engine == "vibevoice":
                            if cache_key != model_key:  # Don't clear the model we're about to load
                                print(f"🔄 VibeVoice: Moving existing model to CPU instead of clearing (allows reuse)")
                                # Force CPU migration instead of deletion
                                try:
                                    wrapper.partially_unload('cpu', wrapper._memory_size)
                                except:
                                    should_clear = True  # Fallback to clearing if CPU migration fails
                        
                        if should_clear:
                            models_to_clear.append(cache_key)
                    
                    if models_to_clear:
                        print(f"🗑️ Clearing {len(models_to_clear)} TTS models to free VRAM")
                        for key in models_to_clear:
                            self.remove_model(key)
                            
            except Exception as e:
                # Silently ignore memory management errors to avoid spam
                pass
        
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
        
        # Ensure device parameter is available to factory function
        factory_kwargs['device'] = device
        model = model_factory_func(**factory_kwargs)
        
        # Calculate memory usage
        memory_size = ComfyUIModelWrapper.calculate_model_memory(model)
        
        # Create model info - for stateless wrappers, use a generic engine name to prevent CUDA graph handling
        actual_engine = engine
        if hasattr(model, '_wrapped_engine') and engine == "higgs_audio":
            # This is a stateless wrapper - use generic name to prevent ComfyUI from doing special CUDA handling
            actual_engine = "stateless_tts"
            print(f"🔒 Treating {engine} stateless wrapper as generic TTS model to avoid CUDA graph interference")
        
        model_info = ModelInfo(
            model=model,
            model_type=model_type,
            engine=actual_engine,  # Use generic name for stateless wrappers
            device=device,
            memory_size=memory_size,
            load_device=device
        )
        
        # Wrap for ComfyUI
        wrapper = ComfyUIModelWrapper(model, model_info)
        
        # Cache the wrapper
        self._model_cache[model_key] = wrapper
        
        # Register with ComfyUI using the proper load_models_gpu method
        if COMFYUI_AVAILABLE and model_management is not None:
            # Try the safer manual approach first since load_models_gpu seems to have issues
            try:
                # Manually add to current_loaded_models using LoadedModel (ComfyUI's internal approach)
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
            
            # Normal destruction for all engines
            self._model_cache.pop(model_key)
            
            # Remove from ComfyUI tracking if available
            if COMFYUI_AVAILABLE and model_management is not None:
                try:
                    if hasattr(model_management, 'current_loaded_models'):
                        if wrapper in model_management.current_loaded_models:
                            model_management.current_loaded_models.remove(wrapper)
                            print(f"🗑️ Removed model from ComfyUI tracking")
                except Exception as e:
                    print(f"⚠️ Failed to remove from ComfyUI tracking: {e}")
            
            # Unload from GPU
            wrapper.model_unload()
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