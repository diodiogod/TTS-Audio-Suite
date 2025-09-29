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

# Import extra paths support
from utils.models.extra_paths import get_all_tts_model_paths, find_model_in_paths

# Import ComfyUI model wrapper for integration
from utils.models.comfyui_model_wrapper import tts_model_manager

# Use ImportManager for robust dependency checking
# Try imports first to populate availability status
tts_success, ChatterboxTTS, tts_source = import_manager.import_chatterbox_tts()
vc_success, ChatterboxVC, vc_source = import_manager.import_chatterbox_vc()
f5tts_success, F5TTS, f5tts_source = import_manager.import_f5tts()

# Try VibeVoice imports
try:
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from engines.vibevoice_engine import VibeVoiceDownloader
    VIBEVOICE_AVAILABLE = True
except ImportError as e:
    VibeVoiceForConditionalGenerationInference = None
    VibeVoiceProcessor = None
    VibeVoiceDownloader = None
    VIBEVOICE_AVAILABLE = False

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
        
        # 2. Check configured TTS model paths (extra_model_paths.yaml aware)
        tts_model_paths = get_all_tts_model_paths('TTS')
        
        for base_tts_path in tts_model_paths:
            comfyui_tts_dir = os.path.join(base_tts_path, "chatterbox")
            if not os.path.exists(comfyui_tts_dir):
                continue
            
            # Check for direct s3gen.pt in chatterbox folder
            direct_model = os.path.join(comfyui_tts_dir, "s3gen.pt")
            if os.path.exists(direct_model):
                model_paths.append(("comfyui", comfyui_tts_dir))
                return model_paths
            
            # Check for language subdirectories (English, German, etc.)
            try:
                for item in os.listdir(comfyui_tts_dir):
                    item_path = os.path.join(comfyui_tts_dir, item)
                    if os.path.isdir(item_path):
                        # Check if this subdirectory contains ChatterBox model files
                        required_files = ["s3gen.", "ve.", "t3_cfg.", "tokenizer.json"]
                        has_model = False
                        for file in os.listdir(item_path):
                            for required in required_files:
                                if file.startswith(required) and (file.endswith(".pt") or file.endswith(".safetensors")):
                                    has_model = True
                                    break
                            if has_model:
                                break
                        if has_model:
                            model_paths.append(("comfyui", comfyui_tts_dir))
                            return model_paths
            except OSError:
                pass
        
        # 3. Check legacy location (direct chatterbox) for backward compatibility
        comfyui_legacy_dir = os.path.join(folder_paths.models_dir, "chatterbox")
        if os.path.exists(comfyui_legacy_dir):
            # Check for direct s3gen.pt in chatterbox folder
            direct_model = os.path.join(comfyui_legacy_dir, "s3gen.pt")
            if os.path.exists(direct_model):
                model_paths.append(("comfyui", comfyui_legacy_dir))
                return model_paths
            
            # Check for language subdirectories
            try:
                for item in os.listdir(comfyui_legacy_dir):
                    item_path = os.path.join(comfyui_legacy_dir, item)
                    if os.path.isdir(item_path):
                        # Check if this subdirectory contains ChatterBox model files
                        required_files = ["s3gen.", "ve.", "t3_cfg.", "tokenizer.json"]
                        has_model = False
                        for file in os.listdir(item_path):
                            for required in required_files:
                                if file.startswith(required) and (file.endswith(".pt") or file.endswith(".safetensors")):
                                    has_model = True
                                    break
                            if has_model:
                                break
                        if has_model:
                            model_paths.append(("comfyui", comfyui_legacy_dir))
                            return model_paths
            except OSError:
                pass
        
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
                os.path.join(folder_paths.models_dir, "TTS", "chatterbox", language),
                os.path.join(folder_paths.models_dir, "TTS", "chatterbox", language.lower()),
                os.path.join(folder_paths.models_dir, "chatterbox", language),  # Legacy
                os.path.join(folder_paths.models_dir, "chatterbox", language.lower()),  # Legacy
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

    def get_model_cache_key(self, model_type: str, device: str, source: str, path: Optional[str] = None, language: str = "English") -> str:
        """
        Generate a cache key for model instances.
        
        Args:
            model_type: Type of model ('tts' or 'vc')
            device: Target device ('cuda', 'cpu')
            source: Model source ('bundled', 'comfyui', 'huggingface')
            path: Optional path for local models
            language: Language model (for multilingual support)
            
        Returns:
            Cache key string
        """
        path_component = path or "default"
        return f"{model_type}_{device}_{source}_{path_component}_{language}"
    
    def load_tts_model(self, device: str = "auto", language: str = "English", force_reload: bool = False) -> Any:
        """
        Load ChatterboxTTS model with ComfyUI-integrated caching and language support.
        
        This method now uses ComfyUI's model management system, enabling automatic
        memory management, "Clear VRAM" button functionality, and proper integration.
        
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
            
        # Use unified model interface for ComfyUI integration
        from utils.models.unified_model_interface import load_tts_model
        
        try:
            model = load_tts_model(
                engine_name="chatterbox",
                model_name=language,
                device=device,
                language=language,
                force_reload=force_reload
            )
            
            # Update instance variables for backward compatibility
            self.tts_model = model
            self.current_device = device
            
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load ChatterBox model via unified interface: {e}")
            # Fallback to original logic if unified interface fails
            pass
        
        # Check if we need to load/reload (including language check)
        if not force_reload and self.tts_model is not None and self.current_device == device:
            # Also check if the current model matches the requested language
            current_cache_key = getattr(self, '_current_tts_cache_key', None)
            if current_cache_key and language in current_cache_key:
                return self.tts_model
        
        # Include language in cache check for fallback logic
        cache_key_base = f"tts_{device}_{language}"
        
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
                    cached_model = self._model_cache[cache_key]
                    self.tts_model = cached_model
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
                
                # Check if this language is supported before calling from_pretrained
                # This prevents the engine's internal fallback from bypassing our cache logic
                from engines.chatterbox.language_models import get_model_config
                if not get_model_config(language) and language != "English":
                    # Language not supported - skip to our fallback logic instead of engine's fallback
                    print(f"📦 Language '{language}' not supported by ChatterBox, skipping to fallback")
                    raise Exception(f"Language '{language}' not supported")
                
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
            
            # First check if English is already loaded in any form
            english_cache_keys = [key for key in self._model_cache.keys() if "English" in key and device in key]
            if english_cache_keys and not force_reload:
                existing_key = english_cache_keys[0]  # Use first found English model
                print(f"💾 Reusing already-loaded English model from cache")
                self.tts_model = self._model_cache[existing_key]
                self.current_device = device
                self._current_tts_cache_key = existing_key
                return self.tts_model
            
            # If no English model in cache, try loading English (check local first)
            try:
                cache_key = f"tts_{device}_English_fallback"
                
                if not force_reload and cache_key in self._model_cache:
                    self.tts_model = self._model_cache[cache_key]
                    self.current_device = device
                    self._current_tts_cache_key = cache_key
                    return self.tts_model
                
                # Try local English first before HuggingFace
                english_local_path = self.find_local_language_model("English")
                if english_local_path:
                    print(f"📁 Loading local English ChatterBox model as fallback from: {english_local_path}")
                    model = ChatterboxTTS.from_local(english_local_path, device)
                else:
                    print(f"📦 Loading English ChatterBox model from HuggingFace as fallback")
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
    
    def load_vc_model(self, device: str = "auto", force_reload: bool = False, language: str = "English") -> Any:
        """
        Load ChatterboxVC model with ComfyUI-integrated caching and language support.
        
        This method now uses ComfyUI's model management system, enabling automatic
        memory management, "Clear VRAM" button functionality, and proper integration.
        
        Args:
            device: Target device ('auto', 'cuda', 'cpu')
            force_reload: Force reload even if cached
            language: Language model to use (English, German, Norwegian)
            
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
            
        # Use unified model interface for ComfyUI integration
        from utils.models.unified_model_interface import load_vc_model
        
        try:
            # Resolve local model path first (same logic as find_local_language_model)
            local_model_path = self.find_local_language_model(language)
            
            model = load_vc_model(
                engine_name="chatterbox",
                model_name=language,
                device=device,
                language=language,
                model_path=local_model_path,  # Pass resolved local path
                force_reload=force_reload
            )
            
            # Update instance variables for backward compatibility
            self.vc_model = model
            self.current_device = device
            self.current_vc_language = language
            
            return model
            
        except Exception as e:
            error_str = str(e)
            print(f"⚠️ Failed to load ChatterBox VC model via unified interface: {e}")
            
            # Re-raise clear errors about unsupported languages instead of falling back
            if "Voice conversion not supported" in error_str:
                raise e  # Re-raise the clear error message
            
            # Only fallback to legacy logic for other types of errors
            pass
        
        # Check if we need to load/reload (include language in check)
        current_language = getattr(self, 'current_vc_language', None)
        if (not force_reload and self.vc_model is not None and 
            self.current_device == device and current_language == language):
            return self.vc_model
        
        # For VC, try language-specific model first (like TTS does)
        local_language_path = self.find_local_language_model(language)
        if local_language_path:
            model_paths = [("comfyui", local_language_path), ("huggingface", None)]
        else:
            # Fall back to generic discovery (includes huggingface fallback)
            model_paths = self.find_chatterbox_models()
        
        model_loaded = False
        last_error = None
        
        for source, path in model_paths:
            try:
                cache_key = self.get_model_cache_key("vc", device, source, path, language)
                
                # Check class-level cache first
                if not force_reload and cache_key in self._model_cache:
                    self.vc_model = self._model_cache[cache_key]
                    self.current_device = device
                    self.current_vc_language = language
                    self._model_sources[cache_key] = source
                    model_loaded = True
                    break
                
                # Load model based on source
                if source in ["bundled", "comfyui"]:
                    # Ensure ChatterboxVC is available before attempting to use it
                    if ChatterboxVC is None:
                        raise RuntimeError(f"ChatterboxVC not available - cannot load local model from {path}")
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Note: Local models currently don't support language selection
                        # They use whatever language they were trained on
                        if language != "English":
                            print(f"⚠️ Local VC model at {path} may not support {language} - using existing model")
                        print(f"🐛 Calling ChatterboxVC.from_local({path}, {device})")
                        model = ChatterboxVC.from_local(path, device)
                        print(f"🐛 from_local returned: {model} (type: {type(model)})")
                elif source == "huggingface":
                    # Ensure ChatterboxVC is available before attempting to use it
                    if ChatterboxVC is None:
                        raise RuntimeError(f"ChatterboxVC not available - cannot load model from HuggingFace")
                    
                    print(f"🐛 Calling ChatterboxVC.from_pretrained({device}, {language})")
                    model = ChatterboxVC.from_pretrained(device, language)
                    print(f"🐛 from_pretrained returned: {model} (type: {type(model)})")
                else:
                    continue
                
                # Cache the loaded model
                self._model_cache[cache_key] = model
                self._model_sources[cache_key] = source
                self.vc_model = model
                self.current_device = device
                self.current_vc_language = language
                model_loaded = True
                print(f"✅ Successfully loaded {language} ChatterBox VC model from {source}")
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
                cache_key = self.get_model_cache_key("vc", device, source, path, getattr(self, 'current_vc_language', 'English'))
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
            "any": CHATTERBOX_TTS_AVAILABLE or CHATTERBOX_VC_AVAILABLE,
            "vibevoice": VIBEVOICE_AVAILABLE
        }
    
    def load_vibevoice_model(self, model_name: str = "vibevoice-1.5B", device: str = "auto", force_reload: bool = False):
        """
        Load VibeVoice model using ComfyUI model wrapper for consistency with other engines.
        
        Args:
            model_name: VibeVoice model name ("vibevoice-1.5B" or "vibevoice-7B")
            device: Target device ('cuda', 'cpu', 'auto')
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, processor) for VibeVoice
            
        Raises:
            ImportError: If VibeVoice is not available
            RuntimeError: If model loading fails
        """
        if not VIBEVOICE_AVAILABLE:
            raise ImportError("VibeVoice not available - check installation")
        
        # Use ComfyUI model manager for consistent behavior
        from utils.models.comfyui_model_wrapper import tts_model_manager
        
        # Resolve auto device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache key for VibeVoice models
        cache_key = f"vibevoice_{model_name}_{device}"
        
        def vibevoice_factory(model_name, device):
            """Factory function to create VibeVoice model and processor"""
            # Get model path (downloads if necessary)
            downloader = VibeVoiceDownloader()
            model_path = downloader.get_model_path(model_name)
            if not model_path:
                raise RuntimeError(f"Failed to get VibeVoice model '{model_name}'")
            
            print(f"🔄 Creating VibeVoice model '{model_name}' on {device}...")
            
            # Load model and processor
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device if device != "auto" else None
            )
            processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Move to device if needed
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            
            # Create container to hold both model and processor
            class VibeVoiceModelContainer:
                def __init__(self, model, processor):
                    self.model = model
                    self.processor = processor
                    
                def to(self, device):
                    """Move both model and processor to device"""
                    if hasattr(self.model, 'to'):
                        self.model.to(device)
                    # Note: processor typically doesn't need device movement
                    return self
                    
                def parameters(self):
                    """Return model parameters for memory calculation"""
                    if hasattr(self.model, 'parameters'):
                        return self.model.parameters()
                    return iter([])
            
            return VibeVoiceModelContainer(model, processor)
        
        # Load using ComfyUI model manager
        wrapper = tts_model_manager.load_model(
            model_factory_func=vibevoice_factory,
            model_key=cache_key,
            model_type="tts",
            engine="vibevoice",
            device=device,
            force_reload=force_reload,
            model_name=model_name
        )
        
        # Extract model and processor from container
        container = wrapper.model
        return container.model, container.processor
    
    def unload_vibevoice_models(self):
        """Unload all VibeVoice models using ComfyUI model wrapper system."""
        from utils.models.comfyui_model_wrapper import tts_model_manager
        
        # Find all VibeVoice models in the ComfyUI model manager
        vibevoice_keys = []
        for key, wrapper in tts_model_manager._model_cache.items():
            if wrapper.model_info.engine == "vibevoice":
                vibevoice_keys.append(key)
        
        # Remove them using the ComfyUI model manager (handles unloading automatically)
        for key in vibevoice_keys:
            tts_model_manager.remove_model(key)
        
        if vibevoice_keys:
            print(f"🧹 Unloaded {len(vibevoice_keys)} VibeVoice models via ComfyUI model manager")


# Global model manager instance
model_manager = ModelManager()