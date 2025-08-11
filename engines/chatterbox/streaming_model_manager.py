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
        """Pre-load all required models for the given languages."""
        required_models = self.get_required_models(language_codes)
        available_languages = get_available_languages()
        
        print(f"🚀 STREAMING: Pre-loading {len(required_models)} models for {len(language_codes)} languages")
        
        for model_name in required_models:
            if model_name in self.preloaded_models:
                print(f"♻️ {model_name} already loaded")
                continue
                
            if model_name not in available_languages:
                print(f"⚠️ {model_name} model not available, using English fallback")
                model_name = 'English'
                
            print(f"📦 Loading {model_name} model...")
            try:
                # CRITICAL FIX: Create completely separate model instances
                # Bypass model manager caching to ensure each language has its own instance
                
                # Create a temporary ModelManager with fresh cache for this specific model
                from utils.models.manager import ModelManager
                temp_manager = ModelManager()
                # Clear any cached models to ensure fresh loading
                temp_manager._model_cache.clear()
                
                # Load with force_reload to bypass any shared state
                model_instance = temp_manager.load_tts_model(device, model_name, force_reload=True)
                
                # Store the model with debugging
                self.preloaded_models[model_name] = model_instance
                print(f"✅ {model_name} model loaded with separate instance")
                print(f"🔍 DEBUG: Stored model ID for {model_name}: {id(model_instance)}")
                
                # Verify all stored models are different
                if len(self.preloaded_models) > 1:
                    print(f"🔍 DEBUG: All stored model IDs: {[(k, id(v)) for k, v in self.preloaded_models.items()]}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                # Fallback to using node's method (will overwrite but at least works)
                model_instance = model_manager.load_tts_model(device, model_name, force_reload=True)
                self.preloaded_models[model_name] = model_instance
                print(f"⚠️ Used fallback loading for {model_name}")
        
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