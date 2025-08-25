"""
F5-TTS Model Manager Extension
Extends the existing ModelManager for F5-TTS specific functionality
"""

import os
import warnings
import torch
import folder_paths
from typing import Optional, List, Tuple, Dict, Any

# Import unified model interface for ComfyUI integration
from utils.models.unified_model_interface import load_tts_model


class F5TTSModelManager:
    """
    F5-TTS model manager extending the existing ModelManager pattern.
    Handles F5-TTS specific model discovery and loading.
    """
    
    # Class-level cache for F5-TTS models
    _f5tts_model_cache: Dict[str, Any] = {}
    _f5tts_model_sources: Dict[str, str] = {}
    
    def __init__(self, node_dir: Optional[str] = None):
        """
        Initialize F5TTSModelManager.
        
        Args:
            node_dir: Optional override for the node directory path
        """
        self.node_dir = node_dir or os.path.dirname(os.path.dirname(__file__))
        self.f5tts_available = False
        self._check_f5tts_availability()
    
    def _check_f5tts_availability(self):
        """Check if F5-TTS is available"""
        try:
            import engines.f5_tts
            self.f5tts_available = True
        except ImportError:
            self.f5tts_available = False
    
    def find_f5tts_models(self, model_name: str = None) -> List[Tuple[str, Optional[str]]]:
        """
        Find F5-TTS model files in order of priority:
        1. ComfyUI models/F5-TTS/ directory (primary location)
        2. ComfyUI models/Checkpoints/F5-TTS/ directory (fallback for user convenience)
        3. HuggingFace download
        
        Args:
            model_name: Optional specific model name to search for
        
        Returns:
            List of tuples containing (source_type, path) in priority order
        """
        model_paths = []
        
        # Search paths in order of priority: TTS first, then legacy
        search_paths = [
            os.path.join(folder_paths.models_dir, "TTS", "F5-TTS"),
            os.path.join(folder_paths.models_dir, "F5-TTS"),  # Legacy
            os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS")  # Legacy
        ]
        
        # 1-2. Check ComfyUI models folders
        for search_path in search_paths:
            if os.path.exists(search_path):
                for item in os.listdir(search_path):
                    item_path = os.path.join(search_path, item)
                    if os.path.isdir(item_path):
                        # If specific model name provided, only check matching folders
                        if model_name and item.lower() != model_name.lower():
                            continue
                        
                        # Check if it contains model files
                        has_model = False
                        for ext in [".safetensors", ".pt"]:
                            model_files = [f for f in os.listdir(item_path) if f.endswith(ext)]
                            if model_files:
                                has_model = True
                                break
                        if has_model:
                            model_paths.append(("comfyui", item_path))
        
        # 3. HuggingFace download as fallback
        model_paths.append(("huggingface", None))
        
        return model_paths
    
    def get_f5tts_model_cache_key(self, model_name: str, device: str, source: str, path: Optional[str] = None) -> str:
        """
        Generate a cache key for F5-TTS model instances.
        
        Args:
            model_name: Name of the F5-TTS model
            device: Target device ('cuda', 'cpu')
            source: Model source ('comfyui', 'huggingface')
            path: Optional path for local models
            
        Returns:
            Cache key string
        """
        path_component = path or "default"
        return f"f5tts_{model_name}_{device}_{source}_{path_component}"
    
    def load_f5tts_model(self, model_name: str = "F5TTS_Base", device: str = "auto", 
                        force_reload: bool = False) -> Any:
        """
        Load F5-TTS model with ComfyUI-integrated caching support.
        
        Args:
            model_name: Name of the F5-TTS model to load
            device: Target device ('auto', 'cuda', 'cpu')
            force_reload: Force reload even if cached
            
        Returns:
            F5-TTS model instance
            
        Raises:
            ImportError: If F5-TTS is not available
            RuntimeError: If model loading fails
        """
        if not self.f5tts_available:
            raise ImportError("F5-TTS not available - check installation")
        
        # Resolve auto device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Try unified model interface for ComfyUI integration
        try:
            model = load_tts_model(
                engine_name="f5tts",
                model_name=model_name,
                device=device,
                force_reload=force_reload
            )
            
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load F5-TTS model via unified interface: {e}")
            print(f"🔄 Falling back to direct loading...")
            # Fall back to original logic if unified interface fails
        
        # Get available model paths for specific model
        model_paths = self.find_f5tts_models(model_name)
        
        model_loaded = False
        last_error = None
        
        # Check if this model should be available locally
        local_model_expected = False
        f5tts_search_paths = [
            os.path.join(folder_paths.models_dir, "TTS", "F5-TTS"),
            os.path.join(folder_paths.models_dir, "F5-TTS"),  # Legacy
            os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS")  # Legacy
        ]
        
        for f5tts_path in f5tts_search_paths:
            if os.path.exists(f5tts_path):
                for item in os.listdir(f5tts_path):
                    if item.lower() == model_name.lower() or model_name.lower() in item.lower():
                        local_model_expected = True
                        break
                if local_model_expected:
                    break
        
        for source, path in model_paths:
            try:
                cache_key = self.get_f5tts_model_cache_key(model_name, device, source, path)
                
                # Check class-level cache first
                if not force_reload and cache_key in self._f5tts_model_cache:
                    model = self._f5tts_model_cache[cache_key]
                    self._f5tts_model_sources[cache_key] = source
                    model_loaded = True
                    break
                
                # Load model based on source
                if source == "comfyui" and path:
                    from engines.f5tts.f5tts import ChatterBoxF5TTS
                    model = ChatterBoxF5TTS.from_local(path, device, model_name)
                elif source == "huggingface":
                    # If a local model was expected but failed, don't fallback to HuggingFace
                    if local_model_expected and last_error:
                        print(f"❌ Model '{model_name}' found in local directory but failed to load")
                        print(f"❌ Fallback to HuggingFace disabled for locally expected models")
                        raise RuntimeError(f"Model '{model_name}' exists locally but failed to load. Please check the model files or use a different model.")
                    
                    # Check if model is available on HuggingFace
                    # This prevents calling from_pretrained with unsupported models, 
                    # which would trigger engine fallback and potentially bypass cache
                    from engines.f5tts.f5tts import F5TTS_MODELS
                    if model_name not in F5TTS_MODELS:
                        print(f"❌ Model '{model_name}' not available on HuggingFace")
                        raise RuntimeError(f"Model '{model_name}' is not available on HuggingFace. Please install it locally or use a different model.")
                    
                    from engines.f5tts.f5tts import ChatterBoxF5TTS
                    model = ChatterBoxF5TTS.from_pretrained(device, model_name)
                else:
                    continue
                
                # Cache the loaded model
                self._f5tts_model_cache[cache_key] = model
                self._f5tts_model_sources[cache_key] = source
                model_loaded = True
                break
                
            except Exception as e:
                print(f"⚠️ Failed to load F5-TTS model from {source}: {str(e)}")
                last_error = e
                continue
        
        if not model_loaded:
            error_msg = f"Failed to load F5-TTS model '{model_name}' from any source"
            if last_error:
                error_msg += f". Last error: {last_error}"
            raise RuntimeError(error_msg)
        
        return self._f5tts_model_cache[cache_key]
    
    def get_f5tts_model_source(self, model_name: str, device: str) -> Optional[str]:
        """
        Get the source of a cached F5-TTS model.
        
        Args:
            model_name: Name of the F5-TTS model
            device: Device the model is loaded on
            
        Returns:
            Model source string or None if not cached
        """
        # Try to find in cache by checking different sources with their paths
        model_paths = self.find_f5tts_models(model_name)
        
        for source, path in model_paths:
            cache_key = self.get_f5tts_model_cache_key(model_name, device, source, path)
            if cache_key in self._f5tts_model_sources:
                return self._f5tts_model_sources[cache_key]
        
        return None
    
    def clear_f5tts_cache(self):
        """Clear F5-TTS model cache."""
        self._f5tts_model_cache.clear()
        self._f5tts_model_sources.clear()
    
    def get_f5tts_model_configs(self) -> List[str]:
        """
        Get available F5-TTS model configurations.
        
        Returns:
            List of available F5-TTS model names
        """
        try:
            from engines.f5tts.f5tts import get_f5tts_models
            return get_f5tts_models()
        except ImportError:
            # Fallback list if F5-TTS not available
            return ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]
    
    @property
    def is_f5tts_available(self) -> bool:
        """
        Check if F5-TTS is available.
        
        Returns:
            True if F5-TTS is available, False otherwise
        """
        return self.f5tts_available


# Global F5-TTS model manager instance
f5tts_model_manager = F5TTSModelManager()