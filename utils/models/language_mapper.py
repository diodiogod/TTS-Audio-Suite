"""
Language Model Mapper - Maps language codes to engine-specific models
Provides centralized language-to-model mapping for F5-TTS and ChatterBox engines
"""

from typing import Dict, List, Optional


class LanguageModelMapper:
    """Maps language codes to engine-specific model names."""
    
    def __init__(self, engine_type: str):
        """
        Initialize language model mapper.
        
        Args:
            engine_type: "f5tts" or "chatterbox"
        """
        self.engine_type = engine_type
        self.mappings = self._load_mappings()
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """
        Map language code to engine-specific model name.
        
        Args:
            lang_code: Language code (e.g., 'en', 'de', 'fr')
            default_model: Default model to use for base language
            
        Returns:
            Model name for the specified language
        """
        engine_mappings = self.mappings.get(self.engine_type, {})
        
        # Check if we should use the default model for this language
        # Only use default model if it's actually for the requested language
        if lang_code == 'en':
            # For English, always use English model regardless of default
            if self.engine_type == 'f5tts':
                return 'F5TTS_v1_Base'  # Use v1 for better quality
            else:  # chatterbox
                return 'English'
        
        # Check if language is supported
        if lang_code in engine_mappings:
            return engine_mappings[lang_code]
        else:
            # Language not supported - show warning and fallback to default
            print(f"⚠️ {self.engine_type.title()}: Language '{lang_code}' not supported, falling back to English model")
            return default_model
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes for current engine."""
        engine_mappings = self.mappings.get(self.engine_type, {})
        return list(engine_mappings.keys())
    
    def is_language_supported(self, lang_code: str) -> bool:
        """Check if a language is supported by current engine."""
        return lang_code in self.get_supported_languages()
    
    @staticmethod
    def _load_mappings() -> Dict[str, Dict[str, str]]:
        """Load language mappings from config."""
        return {
            "f5tts": {
                "en": "F5TTS_Base",  # This will be overridden by default_model
                "de": "F5-DE",       # German
                "es": "F5-ES",       # Spanish
                "fr": "F5-FR",       # French
                "it": "F5-IT",       # Italian
                "jp": "F5-JP",       # Japanese
                "th": "F5-TH",       # Thai
                "pt": "F5-PT-BR",    # Portuguese (Brazil)
            },
            "chatterbox": {
                "en": "English",     # This will be overridden by default_model
                "de": "German",      # German
                "no": "Norwegian",   # Norwegian
                "nb": "Norwegian",   # Norwegian Bokmål
                "nn": "Norwegian",   # Norwegian Nynorsk
            }
        }
    
    def get_all_mappings(self) -> Dict[str, Dict[str, str]]:
        """Get all language mappings for all engines."""
        return self.mappings
    
    def add_language_mapping(self, lang_code: str, model_name: str):
        """
        Add or update a language mapping for current engine.
        
        Args:
            lang_code: Language code
            model_name: Model name for this language
        """
        if self.engine_type not in self.mappings:
            self.mappings[self.engine_type] = {}
        
        self.mappings[self.engine_type][lang_code] = model_name
    
    def remove_language_mapping(self, lang_code: str):
        """
        Remove a language mapping for current engine.
        
        Args:
            lang_code: Language code to remove
        """
        if self.engine_type in self.mappings and lang_code in self.mappings[self.engine_type]:
            del self.mappings[self.engine_type][lang_code]


# Global instances for easy access
f5tts_language_mapper = LanguageModelMapper("f5tts")
chatterbox_language_mapper = LanguageModelMapper("chatterbox")


def get_language_mapper(engine_type: str) -> LanguageModelMapper:
    """
    Get language mapper instance for specified engine.
    
    Args:
        engine_type: "f5tts" or "chatterbox"
        
    Returns:
        LanguageModelMapper instance
    """
    if engine_type == "f5tts":
        return f5tts_language_mapper
    elif engine_type == "chatterbox":
        return chatterbox_language_mapper
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def get_model_for_language(engine_type: str, lang_code: str, default_model: str) -> str:
    """
    Convenience function to get model for language.
    
    Args:
        engine_type: "f5tts" or "chatterbox"
        lang_code: Language code
        default_model: Default model for base language
        
    Returns:
        Model name for the specified language
    """
    mapper = get_language_mapper(engine_type)
    return mapper.get_model_for_language(lang_code, default_model)