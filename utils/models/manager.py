"""
Model Manager - Centralized model loading and caching for ChatterBox Voice
Handles model discovery, loading, and caching across different sources
"""

import os
import warnings
import torch
import folder_paths
from typing import Optional, List, Tuple, Dict, Any
from utils.system.import_manager import import_manager

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
    
    def find_local_language_model(self, language: str) -> Optional[str]:
        """
        Find local ChatterBox model for a specific language.
        
        Args:
            language: Language to find model for
            
        Returns:
            Path to local model directory if found, None otherwise
        """
        # Import language models functionality
        try:
            from engines.chatterbox.language_models import find_local_model_path
            return find_local_model_path(language)
        except ImportError:
            # Fallback: check standard locations manually
            language_paths = [
                os.path.join(folder_paths.models_dir, "chatterbox", language),
                os.path.join(folder_paths.models_dir, "chatterbox", language.lower()),
                os.path.join(self.bundled_models_dir, language),
                os.path.join(self.bundled_models_dir, language.lower())
            ]
            
            for path in language_paths:
                if os.path.exists(path):
                    # Check if it contains the required model files
                    required_files = ["ve.", "t3_cfg.", "s3gen.", "tokenizer.json"]
                    has_all_files = True
                    
                    for required in required_files:
                        found = False
                        for ext in [".pt", ".safetensors"]:
                            if required == "tokenizer.json":
                                if os.path.exists(os.path.join(path, required)):
                                    found = True
                                    break
                            else:
                                if os.path.exists(os.path.join(path, required + ext.replace(".", ""))):
                                    found = True
                                    break
                        if not found:
                            has_all_files = False
                            break
                    
                    if has_all_files:
                        return path
            
            return None

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
    
    def load_tts_model(self, device: str = "auto", language: str = "English", force_reload: bool = False) -> Any:
        """
        Load ChatterboxTTS model with caching and language support.
        
        Args:
            device: Target device ('auto', 'cuda', 'cpu')
            language: Language model to load ('English', 'German', 'Norwegian', etc.)
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
        
        # Include language in cache check
        cache_key_base = f"tts_{device}_{language}"
        
        # Check if we need to load/reload (including language check)
        if not force_reload and self.tts_model is not None and self.current_device == device:
            # Also check if the current model matches the requested language
            current_cache_key = getattr(self, '_current_tts_cache_key', None)
            if current_cache_key and language in current_cache_key:
                return self.tts_model
        
        # For English, also check the original model discovery paths
        if language == "English":
            # Check original model paths first for English (backward compatibility)
            model_paths = self.find_chatterbox_models()
            for source, path in model_paths:
                if source in ["bundled", "comfyui"] and path:
                    try:
                        cache_key = f"{cache_key_base}_local_{path}"
                        
                        # Check class-level cache first
                        if not force_reload and cache_key in self._model_cache:
                            self.tts_model = self._model_cache[cache_key]
                            self.current_device = device
                            self._current_tts_cache_key = cache_key
                            return self.tts_model
                        
                        print(f"📁 Loading local English ChatterBox model from: {path}")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ChatterboxTTS.from_local(path, device)
                        
                        # Cache the loaded model
                        self._model_cache[cache_key] = model
                        self._model_sources[cache_key] = source
                        self.tts_model = model
                        self.current_device = device
                        self._current_tts_cache_key = cache_key
                        return self.tts_model
                        
                    except Exception as e:
                        print(f"⚠️ Failed to load local English model from {path}: {e}")
                        continue
        
        # Try to find local model for the specific language
        local_language_path = self.find_local_language_model(language)
        model_loaded = False
        last_error = None
        
        if local_language_path:
            # Load local language-specific model
            try:
                cache_key = f"{cache_key_base}_local_{local_language_path}"
                
                # Check class-level cache first
                if not force_reload and cache_key in self._model_cache:
                    self.tts_model = self._model_cache[cache_key]
                    self.current_device = device
                    self._current_tts_cache_key = cache_key
                    return self.tts_model
                
                print(f"📁 Loading local {language} ChatterBox model from: {local_language_path}")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ChatterboxTTS.from_local(local_language_path, device)
                
                # Cache the loaded model
                self._model_cache[cache_key] = model
                self._model_sources[cache_key] = "local"
                self.tts_model = model
                self.current_device = device
                self._current_tts_cache_key = cache_key
                model_loaded = True
                
            except Exception as e:
                print(f"⚠️ Failed to load local {language} model: {e}")
                last_error = e
        
        # If local loading failed or no local model, try HuggingFace
        if not model_loaded:
            try:
                cache_key = f"{cache_key_base}_huggingface"
                
                # Check class-level cache first
                if not force_reload and cache_key in self._model_cache:
                    self.tts_model = self._model_cache[cache_key]
                    self.current_device = device
                    self._current_tts_cache_key = cache_key
                    return self.tts_model
                
                print(f"📦 Loading {language} ChatterBox model from HuggingFace")
                model = ChatterboxTTS.from_pretrained(device, language=language)
                
                # Cache the loaded model
                self._model_cache[cache_key] = model
                self._model_sources[cache_key] = "huggingface"
                self.tts_model = model
                self.current_device = device
                self._current_tts_cache_key = cache_key
                model_loaded = True
                
            except Exception as e:
                print(f"⚠️ Failed to load {language} model from HuggingFace: {e}")
                last_error = e
        
        # Fallback: try English if requested language failed and it's not English
        if not model_loaded and language != "English":
            print(f"🔄 Falling back to English model...")
            try:
                cache_key = f"tts_{device}_English_fallback"
                
                if not force_reload and cache_key in self._model_cache:
                    self.tts_model = self._model_cache[cache_key]
                    self.current_device = device
                    self._current_tts_cache_key = cache_key
                    return self.tts_model
                
                model = ChatterboxTTS.from_pretrained(device, language="English")
                
                # Cache the loaded model
                self._model_cache[cache_key] = model
                self._model_sources[cache_key] = "huggingface"
                self.tts_model = model
                self.current_device = device
                self._current_tts_cache_key = cache_key
                model_loaded = True
                
            except Exception as e:
                print(f"❌ Even English fallback failed: {e}")
                last_error = e
        
        if not model_loaded:
            error_msg = f"Failed to load ChatterBox TTS model for language '{language}' from any source"
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
            # Use the current cache key to determine source
            current_cache_key = getattr(self, '_current_tts_cache_key', None)
            if current_cache_key:
                # Extract source from cache key or _model_sources
                source = self._model_sources.get(current_cache_key)
                if source:
                    return source
                
                # Fallback: parse from cache key format
                if "_local_" in current_cache_key:
                    return "comfyui"
                elif "_huggingface" in current_cache_key:
                    return "huggingface"
                elif "_fallback" in current_cache_key:
                    return "huggingface (fallback)"
            
            # Legacy fallback
            device = self.current_device or "cpu"
            model_paths = self.find_chatterbox_models()
            if model_paths:
                source, path = model_paths[0]
                return source
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