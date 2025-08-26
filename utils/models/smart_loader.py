"""
Universal Smart Model Loader
Provides intelligent model loading that works across all TTS engines.
Prevents unnecessary model reloading and enables cross-engine resource sharing.
"""

import hashlib
import torch
from typing import Any, Callable, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a loaded model instance."""
    model_instance: Any
    engine_type: str
    model_name: str
    device: str
    model_id: str  # Unique identifier for the model


class SmartModelLoader:
    """
    Universal smart model loading that works across all TTS engines.
    
    Features:
    - Prevents unnecessary model reloading within same engine
    - Enables cross-engine model sharing when possible
    - Provides consistent logging across all engines
    - Future-proof design for new engines
    """
    
    def __init__(self):
        # Global cache of loaded models across all engines
        self._global_model_cache: Dict[str, ModelInfo] = {}
        
        # Engine-specific current model tracking
        self._current_models: Dict[str, Dict[str, Any]] = {}
    
    def _generate_cache_key(self, engine_type: str, model_name: str, device: str) -> str:
        """Generate unique cache key for model identification by language."""
        key_data = f"{engine_type}_{model_name}_{device}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _is_same_model(self, current_model: Any, model_name: str, device: str, engine_type: str) -> bool:
        """Check if current model matches requested model."""
        if current_model is None:
            return False
        
        # Check if the current model instance matches a cached model for this language
        cache_key = self._generate_cache_key(engine_type, model_name, device)
        if cache_key in self._global_model_cache:
            cached_info = self._global_model_cache[cache_key]
            return cached_info.model_instance is current_model
        
        return False
    
    def _check_cross_engine_cache(self, engine_type: str, model_name: str, device: str) -> Optional[Any]:
        """Check if model exists in global cache from any engine."""
        cache_key = self._generate_cache_key(engine_type, model_name, device)
        
        if cache_key in self._global_model_cache:
            model_info = self._global_model_cache[cache_key]
            print(f"♻️ SMART LOADER: Reusing {model_name} model from global cache (engine: {model_info.engine_type}, ID: {id(model_info.model_instance)})")
            
            # CRITICAL: Ensure model is on correct device after cache reuse
            model_instance = model_info.model_instance
            
            # Convert "auto" to actual device
            actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            
            if hasattr(model_instance, 'to'):
                try:
                    model_instance.to(actual_device)
                except Exception as e:
                    print(f"⚠️ SMART LOADER: Failed to move model to {actual_device}: {e}")
            
            # Also move nested components recursively (like ComfyUI wrapper does)
            self._ensure_all_components_on_device(model_instance, actual_device)
            
            return model_instance
        
        return None
    
    def _ensure_all_components_on_device(self, obj, target_device: str, depth: int = 0, max_depth: int = 3):
        """
        Recursively ensure all PyTorch components are on the target device.
        Similar to ComfyUI wrapper's device management.
        """
        if depth > max_depth or obj is None:
            return
            
        # Move PyTorch modules
        if hasattr(obj, 'to') and hasattr(obj, 'parameters') and callable(getattr(obj, 'to')):
            try:
                obj.to(target_device)
            except Exception:
                pass
        
        # Recurse through object attributes
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if not attr_name.startswith('_') and attr_value is not None:
                    # Skip problematic attributes
                    if attr_name in ['_modules', '_parameters', '_buffers']:
                        continue
                    try:
                        self._ensure_all_components_on_device(attr_value, target_device, depth + 1, max_depth)
                    except Exception:
                        pass
    
    def _update_caches(self, engine_type: str, model_name: str, device: str, model_instance: Any) -> None:
        """Update both global cache and engine-specific tracking."""
        cache_key = self._generate_cache_key(engine_type, model_name, device)
        
        # Update global cache
        model_info = ModelInfo(
            model_instance=model_instance,
            engine_type=engine_type,
            model_name=model_name,
            device=device,
            model_id=f"{engine_type}_{model_name}_{id(model_instance)}"
        )
        self._global_model_cache[cache_key] = model_info
        
        # Update engine-specific tracking
        if engine_type not in self._current_models:
            self._current_models[engine_type] = {}
        
        self._current_models[engine_type]['current'] = {
            'model_name': model_name,
            'device': device,
            'instance': model_instance
        }
    
    def load_model_if_needed(self, 
                           engine_type: str, 
                           model_name: str, 
                           current_model: Any, 
                           device: str, 
                           load_callback: Callable[[str, str], Any],
                           force_reload: bool = False) -> Tuple[Any, bool]:
        """
        Universal smart model loading logic.
        
        Args:
            engine_type: Engine identifier (e.g., 'chatterbox', 'f5tts')
            model_name: Name/identifier of the model to load
            current_model: Currently loaded model instance (if any)
            device: Target device ('cuda', 'cpu', etc.)
            load_callback: Function to load model - signature: (device, model_name) -> model
            force_reload: Force loading even if model appears to be loaded
            
        Returns:
            Tuple of (model_instance, was_already_loaded)
        """
        if force_reload:
            print(f"🔄 SMART LOADER: Force reloading {model_name} for {engine_type}")
        else:
            # Check if requested model is already the current model
            if self._is_same_model(current_model, model_name, device, engine_type):
                print(f"♻️ SMART LOADER: {model_name} already loaded for {engine_type} (ID: {id(current_model)})")
                return current_model, True
            
            # Check shared caches across engines
            cached_model = self._check_cross_engine_cache(engine_type, model_name, device)
            if cached_model:
                # Update engine-specific tracking to reflect the reused model
                self._current_models.setdefault(engine_type, {})['current'] = {
                    'model_name': model_name,
                    'device': device,
                    'instance': cached_model
                }
                return cached_model, True
        
        # Load new model using provided callback
        try:
            new_model = load_callback(device, model_name)
            self._update_caches(engine_type, model_name, device, new_model)
            return new_model, False
        except Exception as e:
            print(f"❌ SMART LOADER: Failed to load {model_name} for {engine_type}: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached models."""
        stats = {
            'total_cached_models': len(self._global_model_cache),
            'engines_with_models': len(self._current_models),
            'cached_models_by_engine': {}
        }
        
        for cache_key, model_info in self._global_model_cache.items():
            engine = model_info.engine_type
            if engine not in stats['cached_models_by_engine']:
                stats['cached_models_by_engine'][engine] = []
            
            stats['cached_models_by_engine'][engine].append({
                'model_name': model_info.model_name,
                'device': model_info.device,
                'model_id': id(model_info.model_instance)
            })
        
        return stats
    
    def clear_cache(self, engine_type: Optional[str] = None) -> None:
        """Clear model cache for specific engine or all engines."""
        if engine_type:
            # Clear specific engine
            keys_to_remove = [k for k, v in self._global_model_cache.items() 
                            if v.engine_type == engine_type]
            for key in keys_to_remove:
                del self._global_model_cache[key]
            
            if engine_type in self._current_models:
                del self._current_models[engine_type]
            
            print(f"🧹 SMART LOADER: Cleared cache for {engine_type}")
        else:
            # Clear all
            self._global_model_cache.clear()
            self._current_models.clear()
            print("🧹 SMART LOADER: Cleared all model caches")


# Global instance for use across all engines
smart_model_loader = SmartModelLoader()