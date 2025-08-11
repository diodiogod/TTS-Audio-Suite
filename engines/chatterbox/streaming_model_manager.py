"""
Streaming Model Manager for ChatterBox
Handles pre-loading and efficient switching of multiple language models for streaming workers.
"""

from typing import Dict, Set, List, Optional
from engines.chatterbox.language_models import get_available_languages, get_model_config
from utils.models.language_mapper import get_language_mapper

class StreamingModelManager:
    """
    Manages multiple pre-loaded language models for efficient streaming processing.
    Allows workers to access any language model without switching delays.
    Uses the centralized language mapping system.
    """
    
    def __init__(self, default_language: str = "English"):
        self.preloaded_models: Dict[str, any] = {}
        self.default_language = default_language
        self.language_mapper = get_language_mapper("chatterbox")
    
    def get_required_models(self, language_codes: List[str]) -> Set[str]:
        """Get set of model names needed for given language codes using central mapper."""
        models = set()
        for lang_code in language_codes:
            model_name = self.language_mapper.get_model_for_language(lang_code, self.default_language)
            models.add(model_name)
        return models
    
    def preload_models(self, language_codes: List[str], model_manager, device: str) -> None:
        """Pre-load all required models for the given languages, reusing existing models when possible."""
        required_models = self.get_required_models(language_codes)
        available_languages = get_available_languages()
        
        print(f"🚀 STREAMING: Pre-loading {len(required_models)} models for {len(language_codes)} languages")
        
        for model_name in required_models:
            if model_name in self.preloaded_models:
                print(f"♻️ {model_name} already loaded in streaming cache")
                continue
                
            if model_name not in available_languages:
                print(f"⚠️ {model_name} model not available, using English fallback")
                model_name = 'English'
            
            # SMART REUSE: Check if main ModelManager already has this model loaded
            model_found_in_cache = False
            for cache_key, cached_model in model_manager._model_cache.items():
                # Check if cache key contains our model name and device
                if (model_name.lower() in cache_key.lower() and 
                    device in cache_key and 
                    cached_model is not None):
                    print(f"♻️ Reusing {model_name} from main ModelManager cache (ID: {id(cached_model)})")
                    self.preloaded_models[model_name] = cached_model
                    model_found_in_cache = True
                    break
            
            # Only load if not found in existing cache
            if not model_found_in_cache:
                print(f"📦 Loading {model_name} model...")
                try:
                    # Use the main model manager to load (this will cache it properly)
                    model_instance = model_manager.load_tts_model(device, model_name)
                    
                    # Store reference in our streaming cache
                    self.preloaded_models[model_name] = model_instance
                    print(f"✅ {model_name} model loaded and cached (ID: {id(model_instance)})")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    # Try fallback to English if not already English
                    if model_name != 'English' and 'English' in self.preloaded_models:
                        print(f"🔄 Using English model as fallback for {model_name}")
                        self.preloaded_models[model_name] = self.preloaded_models['English']
                    else:
                        print(f"❌ No fallback available for {model_name}")
            
            # Debug: Show all current model IDs
            if len(self.preloaded_models) > 1:
                print(f"🔍 DEBUG: All stored model IDs: {[(k, id(v)) for k, v in self.preloaded_models.items()]}")
        
        print(f"🚀 Model pre-loading complete! {len(self.preloaded_models)} models ready")
    
    def get_model_for_language(self, language_code: str, fallback_model=None):
        """Get the appropriate pre-loaded model for a language code."""
        model_name = self.language_mapper.get_model_for_language(language_code, 'English')
        if model_name in self.preloaded_models:
            returned_model = self.preloaded_models[model_name]
            print(f"✅ Using preloaded '{model_name}' model for '{language_code}' language")
            print(f"🔍 DEBUG: Returning model ID for {model_name}: {id(returned_model)}")
            return returned_model
        elif 'English' in self.preloaded_models:
            print(f"⚠️ Fallback: Using English model for language '{language_code}' (requested '{model_name}')")
            return self.preloaded_models['English']  # Fallback
        else:
            print(f"❌ No model found for language '{language_code}', using fallback")
            return fallback_model  # Use provided fallback
    
    def cleanup(self):
        """Clean up pre-loaded models to free memory."""
        print(f"🧹 Cleaning up {len(self.preloaded_models)} pre-loaded models")
        self.preloaded_models.clear()