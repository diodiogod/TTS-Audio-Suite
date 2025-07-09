"""
F5-TTS Model Manager Extension
Extends the existing ModelManager for F5-TTS specific functionality
"""

import os
import warnings
import torch
import folder_paths
from typing import Optional, List, Tuple, Dict, Any


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
            import f5_tts
            self.f5tts_available = True
        except ImportError:
            self.f5tts_available = False
    
    def find_f5tts_models(self) -> List[Tuple[str, Optional[str]]]:
        """
        Find F5-TTS model files in order of priority:
        1. ComfyUI models/F5-TTS/ directory
        2. HuggingFace download
        
        Returns:
            List of tuples containing (source_type, path) in priority order
        """
        model_paths = []
        
        # 1. Check ComfyUI models folder - F5-TTS directory
        comfyui_f5tts_path = os.path.join(folder_paths.models_dir, "F5-TTS")
        if os.path.exists(comfyui_f5tts_path):
            for item in os.listdir(comfyui_f5tts_path):
                item_path = os.path.join(comfyui_f5tts_path, item)
                if os.path.isdir(item_path):
                    # Check if it contains model files
                    has_model = False
                    for ext in [".safetensors", ".pt"]:
                        model_files = [f for f in os.listdir(item_path) if f.endswith(ext)]
                        if model_files:
                            has_model = True
                            break
                    if has_model:
                        model_paths.append(("comfyui", item_path))
        
        # 2. HuggingFace download as fallback
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
        Load F5-TTS model with caching support.
        
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
        
        # Get available model paths
        model_paths = self.find_f5tts_models()
        
        model_loaded = False
        last_error = None
        
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
                    from chatterbox.f5tts import ChatterBoxF5TTS
                    model = ChatterBoxF5TTS.from_local(path, device, model_name)
                elif source == "huggingface":
                    from chatterbox.f5tts import ChatterBoxF5TTS
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
        # Try to find in cache by checking different sources
        for source in ["comfyui", "huggingface"]:
            cache_key = self.get_f5tts_model_cache_key(model_name, device, source)
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
            from chatterbox.f5tts.f5tts import get_f5tts_models
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