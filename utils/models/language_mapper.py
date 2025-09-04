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
            lang_code: Language code (e.g., 'en', 'de', 'fr') or local model (e.g., 'local:German')
            default_model: Default model to use for base language
            
        Returns:
            Model name for the specified language
        """
        # Handle local models - normalize to base model name
        if lang_code.startswith('local:'):
            return lang_code[6:]  # Remove "local:" prefix - they use same model as base language
        
        engine_mappings = self.mappings.get(self.engine_type, {})
        
        # Check if we should use the default model for this language
        # Only use default model if it's actually for the requested language
        if lang_code == 'en':
            # For English, prefer the default model if it's an English model
            if self.engine_type == 'f5tts':
                # Check if default model is already an English F5-TTS model
                english_models = ['F5TTS_Base', 'F5TTS_v1_Base', 'E2TTS_Base']
                if default_model in english_models:
                    return default_model  # Use engine's configured model
                else:
                    return 'F5TTS_v1_Base'  # Use v1 for better quality as fallback
            elif self.engine_type == 'chatterbox':
                return 'English'
            elif self.engine_type == 'vibevoice':
                # VibeVoice uses same model for both EN/ZH, so use configured model
                vibevoice_models = ['vibevoice-1.5B', 'vibevoice-7B']
                if default_model in vibevoice_models:
                    return default_model  # Use engine's configured model
                else:
                    return 'vibevoice-1.5B'  # Default fallback
        
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
        # Dynamic ChatterBox language mappings
        chatterbox_mappings = LanguageModelMapper._get_dynamic_chatterbox_mappings()
        
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
                "pt-br": "F5-PT-BR", # Portuguese (Brazil) - alternative format
                "hi": "F5-Hindi-Small",  # Hindi - uses Small model from IIT Madras
                # Note: Other Indian languages (as, bn, gu, kn, ml, mr, or, pa, ta, te) fall back to base F5TTS models
                # IndicF5 was removed due to architecture incompatibility
            },
            "chatterbox": chatterbox_mappings,
            "vibevoice": {
                "en": "vibevoice-1.5B",  # This will be overridden by default_model
                "zh": "vibevoice-1.5B",  # Chinese - same model supports both EN/ZH
                "zh-cn": "vibevoice-1.5B",  # Simplified Chinese
                "chinese": "vibevoice-1.5B",  # Alternative format
                # VibeVoice models support both English and Chinese with the same model
            }
        }
    
    @staticmethod
    def _get_dynamic_chatterbox_mappings() -> Dict[str, str]:
        """
        Generate dynamic ChatterBox language mappings from the language registry.
        Maps language codes to ChatterBox model names.
        """
        try:
            from engines.chatterbox.language_models import CHATTERBOX_MODELS
            
            # Create mappings from language codes to model names
            mappings = {}
            
            # Map canonical language codes (from character_parser alias resolution) to ChatterBox models
            # Character parser handles alias resolution: [Brasil:] -> 'pt-br', [USA:] -> 'en', etc.
            # This maps the resolved canonical codes to actual model names
            language_mappings = {
                # Canonical codes to ChatterBox models
                "en": "English",                # [USA:], [America:], [English:] -> en -> English
                "de": "German",                 # [German:], [Deutschland:] -> de -> German  
                "no": "Norwegian",              # [Norway:], [Norsk:] -> no -> Norwegian
                "nb": "Norwegian",              # Norwegian Bokmål
                "nn": "Norwegian",              # Norwegian Nynorsk
                "fr": "French",                 # [France:], [Français:] -> fr -> French
                "ru": "Russian",                # [Russia:], [русский:] -> ru -> Russian
                "hy": "Armenian",               # Armenian
                "ka": "Georgian",               # Georgian  
                "ja": "Japanese",               # [Japan:], [日本語:] -> ja -> Japanese
                "ko": "Korean",                 # [Korea:], [한국어:] -> ko -> Korean
                "it": "Italian",                # [Italy:], [Italia:] -> it -> Italian
                
                # ChatterBox-specific model variants (these bypass character_parser aliases)
                "de-expressive": "German (SebastianBodza)",    # Direct model selection
                "de-kartoffel": "German (SebastianBodza)",     # Direct model selection
                "de-multi": "German (havok2)",                 # Direct model selection
                "de-hybrid": "German (havok2)",                # Direct model selection 
                "de-best": "German (havok2)",                  # Direct model selection - user rated best
                
                # Future expansion when we get Portuguese models:
                # "pt-br": "Portuguese (Brazil)",  # [Brasil:], [BR:] -> pt-br -> Portuguese (Brazil)
                # "pt-pt": "Portuguese (Portugal)", # [Portugal:] -> pt-pt -> Portuguese (Portugal)
            }
            
            # Only add mappings for models that actually exist in registry
            for lang_code, model_name in language_mappings.items():
                if model_name in CHATTERBOX_MODELS:
                    mappings[lang_code] = model_name
            
            return mappings
            
        except ImportError:
            # Fallback to static mappings if ChatterBox not available
            return {
                "en": "English",
                "de": "German", 
                "no": "Norwegian",
                "nb": "Norwegian",
                "nn": "Norwegian",
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
vibevoice_language_mapper = LanguageModelMapper("vibevoice")


def get_language_mapper(engine_type: str) -> LanguageModelMapper:
    """
    Get language mapper instance for specified engine.
    
    Args:
        engine_type: "f5tts", "chatterbox", or "vibevoice"
        
    Returns:
        LanguageModelMapper instance
    """
    if engine_type == "f5tts":
        return f5tts_language_mapper
    elif engine_type == "chatterbox":
        return chatterbox_language_mapper
    elif engine_type == "vibevoice":
        return vibevoice_language_mapper
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def get_model_for_language(engine_type: str, lang_code: str, default_model: str) -> str:
    """
    Convenience function to get model for language.
    
    Args:
        engine_type: "f5tts", "chatterbox", or "vibevoice"
        lang_code: Language code
        default_model: Default model for base language
        
    Returns:
        Model name for the specified language
    """
    mapper = get_language_mapper(engine_type)
    return mapper.get_model_for_language(lang_code, default_model)