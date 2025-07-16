"""
Model Manager - Centralized model loading and caching for ChatterBox Voice
Handles model discovery, loading, and caching across different sources
"""

import os
import warnings
import torch
import folder_paths
from typing import Optional, List, Tuple, Dict, Any
from .import_manager import import_manager

# Use ImportManager for robust dependency checking
# Try imports first to populate availability status
tts_success, ChatterboxTTS, tts_source = import_manager.import_chatterbox_tts()
vc_success, ChatterboxVC, vc_source = import_manager.import_chatterbox_vc()
f5tts_success, F5TTS, f5tts_source = import_manager.import_f5tts()

# Set availability flags
CHATTERBOX_TTS_AVAILABLE = tts_success
CHATTERBOX_VC_AVAILABLE = vc_success
F5TTS_AVAILABLE = f5tts_success
USING_BUNDLED_CHATTERBOX = tts_source == "bundled" or vc_source == "bundled"


class ModelManager:
    """
    Centralized model loading and caching manager for ChatterBox Voice.
    Handles model discovery, loading from different sources, and caching.
    """
    
    # Class-level cache for shared model instances
    _model_cache: Dict[str, Any] = {}
    _model_sources: Dict[str, str] = {}
    
    def __init__(self, node_dir: Optional[str] = None):
        """
        Initialize ModelManager with optional node directory override.
        
        Args:
            node_dir: Optional override for the node directory path
        """
        self.node_dir = node_dir or os.path.dirname(os.path.dirname(__file__))
        self.bundled_chatterbox_dir = os.path.join(self.node_dir, "chatterbox")
        self.bundled_models_dir = os.path.join(self.node_dir, "models", "chatterbox")
        
        # Instance-level model references
        self.tts_model: Optional[Any] = None
        self.vc_model: Optional[Any] = None
        self.current_device: Optional[str] = None
    
    def find_chatterbox_models(self) -> List[Tuple[str, Optional[str]]]:
        """
        Find ChatterBox model files in order of priority.
        
        Returns:
            List of tuples containing (source_type, path) in priority order:
            - bundled: Models bundled with the extension
            - comfyui: Models in ComfyUI models directory
            - huggingface: Download from Hugging Face (path is None)
        """
        model_paths = []
        
        # 1. Check for bundled models in node folder
        bundled_model_path = os.path.join(self.bundled_models_dir, "s3gen.pt")
        if os.path.exists(bundled_model_path):
            model_paths.append(("bundled", self.bundled_models_dir))
            return model_paths  # Return immediately if bundled models found
        
        # 2. Check ComfyUI models folder - standard location
        comfyui_model_path_standard = os.path.join(folder_paths.models_dir, "chatterbox", "s3gen.pt")
        if os.path.exists(comfyui_model_path_standard):
            model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_standard)))
            return model_paths
        
        # 3. Check legacy location (TTS/chatterbox) for backward compatibility
        comfyui_model_path_legacy = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", "s3gen.pt")
        if os.path.exists(comfyui_model_path_legacy):
            model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_legacy)))
            return model_paths
        
        # 4. HuggingFace download as fallback
        model_paths.append(("huggingface", None))
        
        return model_paths
    
    def get_model_cache_key(self, model_type: str, device: str, source: str, path: Optional[str] = None) -> str:
        """
        Generate a cache key for model instances.
        
        Args:
            model_type: Type of model ('tts' or 'vc')
            device: Target device ('cuda', 'cpu')
            source: Model source ('bundled', 'comfyui', 'huggingface')
            path: Optional path for local models
            
        Returns:
            Cache key string
        """
        path_component = path or "default"
        return f"{model_type}_{device}_{source}_{path_component}"
    
    def load_tts_model(self, device: str = "auto", force_reload: bool = False) -> Any:
        """
        Load ChatterboxTTS model with caching.
        
        Args:
            device: Target device ('auto', 'cuda', 'cpu')
            force_reload: Force reload even if cached
            
        Returns:
            ChatterboxTTS model instance
            
        Raises:
            ImportError: If ChatterboxTTS is not available
            RuntimeError: If model loading fails
        """
        if not CHATTERBOX_TTS_AVAILABLE:
            raise ImportError("ChatterboxTTS not available - check installation or add bundled version")
        
        # Resolve auto device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if we need to load/reload
        if not force_reload and self.tts_model is not None and self.current_device == device:
            return self.tts_model
        
        # Get available model paths
        model_paths = self.find_chatterbox_models()
        
        model_loaded = False
        last_error = None
        
        for source, path in model_paths:
            try:
                cache_key = self.get_model_cache_key("tts", device, source, path)
                
                # Check class-level cache first
                if not force_reload and cache_key in self._model_cache:
                    self.tts_model = self._model_cache[cache_key]
                    self.current_device = device
                    self._model_sources[cache_key] = source
                    model_loaded = True
                    break
                
                # Load model based on source
                if source in ["bundled", "comfyui"]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ChatterboxTTS.from_local(path, device)
                elif source == "huggingface":
                    model = ChatterboxTTS.from_pretrained(device)
                else:
                    continue
                
                # Cache the loaded model
                self._model_cache[cache_key] = model
                self._model_sources[cache_key] = source
                self.tts_model = model
                self.current_device = device
                model_loaded = True
                break
                
            except Exception as e:
                print(f"⚠️ Failed to load TTS model from {source}: {str(e)}")
                last_error = e
                continue
        
        if not model_loaded:
            error_msg = f"Failed to load ChatterboxTTS from any source"
            if last_error:
                error_msg += f". Last error: {last_error}"
            raise RuntimeError(error_msg)
        
        return self.tts_model
    
    def load_vc_model(self, device: str = "auto", force_reload: bool = False) -> Any:
        """
        Load ChatterboxVC model with caching.
        
        Args:
            device: Target device ('auto', 'cuda', 'cpu')
            force_reload: Force reload even if cached
            
        Returns:
            ChatterboxVC model instance
            
        Raises:
            ImportError: If ChatterboxVC is not available
            RuntimeError: If model loading fails
        """
        if not CHATTERBOX_VC_AVAILABLE:
            raise ImportError("ChatterboxVC not available - check installation or add bundled version")
        
        # Resolve auto device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if we need to load/reload
        if not force_reload and self.vc_model is not None and self.current_device == device:
            return self.vc_model
        
        # Get available model paths
        model_paths = self.find_chatterbox_models()
        
        model_loaded = False
        last_error = None
        
        for source, path in model_paths:
            try:
                cache_key = self.get_model_cache_key("vc", device, source, path)
                
                # Check class-level cache first
                if not force_reload and cache_key in self._model_cache:
                    self.vc_model = self._model_cache[cache_key]
                    self.current_device = device
                    self._model_sources[cache_key] = source
                    model_loaded = True
                    break
                
                # Load model based on source
                if source in ["bundled", "comfyui"]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ChatterboxVC.from_local(path, device)
                elif source == "huggingface":
                    model = ChatterboxVC.from_pretrained(device)
                else:
                    continue
                
                # Cache the loaded model
                self._model_cache[cache_key] = model
                self._model_sources[cache_key] = source
                self.vc_model = model
                self.current_device = device
                model_loaded = True
                break
                
            except Exception as e:
                last_error = e
                continue
        
        if not model_loaded:
            error_msg = f"Failed to load ChatterboxVC from any source"
            if last_error:
                error_msg += f". Last error: {last_error}"
            raise RuntimeError(error_msg)
        
        return self.vc_model
    
    def get_model_source(self, model_type: str) -> Optional[str]:
        """
        Get the source of the currently loaded model.
        
        Args:
            model_type: Type of model ('tts' or 'vc')
            
        Returns:
            Model source string or None if no model loaded
        """
        if model_type == "tts" and self.tts_model is not None:
            device = self.current_device or "cpu"
            model_paths = self.find_chatterbox_models()
            if model_paths:
                source, path = model_paths[0]
                cache_key = self.get_model_cache_key("tts", device, source, path)
                return self._model_sources.get(cache_key)
        elif model_type == "vc" and self.vc_model is not None:
            device = self.current_device or "cpu"
            model_paths = self.find_chatterbox_models()
            if model_paths:
                source, path = model_paths[0]
                cache_key = self.get_model_cache_key("vc", device, source, path)
                return self._model_sources.get(cache_key)
        
        return None
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear model cache.
        
        Args:
            model_type: Optional model type to clear ('tts', 'vc'), or None for all
        """
        if model_type is None:
            # Clear all
            self._model_cache.clear()
            self._model_sources.clear()
            self.tts_model = None
            self.vc_model = None
            self.current_device = None
        elif model_type == "tts":
            # Clear TTS models
            keys_to_remove = [k for k in self._model_cache.keys() if k.startswith("tts_")]
            for key in keys_to_remove:
                self._model_cache.pop(key, None)
                self._model_sources.pop(key, None)
            self.tts_model = None
        elif model_type == "vc":
            # Clear VC models
            keys_to_remove = [k for k in self._model_cache.keys() if k.startswith("vc_")]
            for key in keys_to_remove:
                self._model_cache.pop(key, None)
                self._model_sources.pop(key, None)
            self.vc_model = None
    
    @property
    def is_available(self) -> Dict[str, bool]:
        """
        Check availability of ChatterBox components.
        
        Returns:
            Dictionary with availability status
        """
        return {
            "tts": CHATTERBOX_TTS_AVAILABLE,
            "vc": CHATTERBOX_VC_AVAILABLE,
            "bundled": USING_BUNDLED_CHATTERBOX,
            "any": CHATTERBOX_TTS_AVAILABLE or CHATTERBOX_VC_AVAILABLE
        }


# Global model manager instance
model_manager = ModelManager()