"""
Streaming Model Manager for ChatterBox
Handles pre-loading and efficient switching of multiple language models for streaming workers.
Now provides stateless wrappers for thread-safe parallel generation.
"""

from typing import Dict, Set, List, Optional
from engines.chatterbox.language_models import get_available_languages, get_model_config
from utils.models.language_mapper import get_language_mapper
from engines.chatterbox.stateless_wrapper import StatelessChatterBoxWrapper

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
        """Pre-load all required models for the given languages using universal smart loader."""
        required_models = self.get_required_models(language_codes)
        available_languages = get_available_languages()
        
        print(f"🚀 STREAMING: Pre-loading {len(required_models)} models for {len(language_codes)} languages")
        
        # Use universal smart model loader for consistency
        from utils.models.smart_loader import smart_model_loader
        
        for model_name in required_models:
            if model_name in self.preloaded_models:
                print(f"♻️ {model_name} already loaded in streaming cache")
                continue
                
            if model_name not in available_languages:
                print(f"⚠️ {model_name} model not available, using English fallback")
                model_name = 'English'
            
            # Use smart loader to get or load the model
            try:
                model_instance, was_cached = smart_model_loader.load_model_if_needed(
                    engine_type="chatterbox",
                    model_name=model_name,
                    current_model=self.preloaded_models.get(model_name),
                    device=device,
                    load_callback=lambda d, m: model_manager.load_tts_model(d, m)
                )
                
                # Store reference in our streaming cache
                self.preloaded_models[model_name] = model_instance
                
                if was_cached:
                    print(f"♻️ STREAMING: Reused {model_name} from smart loader (ID: {id(model_instance)})")
                else:
                    print(f"✅ STREAMING: Loaded {model_name} via smart loader (ID: {id(model_instance)})")
                    
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
    
    def get_stateless_model_for_language(self, language_code: str, fallback_model=None):
        """
        Get a stateless wrapper for the appropriate pre-loaded model.
        This ensures thread-safe parallel generation without state corruption.
        
        Args:
            language_code: Language code (e.g., 'en', 'de', 'fr')
            fallback_model: Fallback model if language not found
            
        Returns:
            StatelessChatterBoxWrapper instance or None
        """
        base_model = self.get_model_for_language(language_code, fallback_model)
        if base_model:
            # Return a stateless wrapper for thread-safe generation
            stateless_wrapper = StatelessChatterBoxWrapper(base_model)
            print(f"🔒 Providing stateless wrapper for '{language_code}' (thread-safe)")
            return stateless_wrapper
        return None
    
    def load_vc_model(self, device: str = "auto", force_reload: bool = False, language: str = "English"):
        """
        Voice Conversion not supported in StreamingModelManager.
        This method exists to prevent signature errors when VC nodes try to call it.
        """
        raise NotImplementedError(
            "Voice Conversion is not supported with StreamingModelManager. "
            "StreamingModelManager is for TTS only. VC requires the main ModelManager from utils.models.manager."
        )
    
    def cleanup(self):
        """Clean up pre-loaded models to free memory."""
        print(f"🧹 Cleaning up {len(self.preloaded_models)} pre-loaded models")
        self.preloaded_models.clear()