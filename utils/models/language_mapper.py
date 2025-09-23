"""
Language Model Mapper - Maps language codes to engine-specific models
Provides centralized language-to-model mapping for F5-TTS and ChatterBox engines
"""

from typing import Dict, List, Optional


# Global language alias system - maps various language names/codes to canonical codes
# Complete copy from character_parser.py language aliases
LANGUAGE_ALIASES = {
    # German variations
    'de': 'de', 'german': 'de', 'deutsch': 'de', 'germany': 'de', 'deutschland': 'de',
    
    # English variations
    'en': 'en', 'english': 'en', 'eng': 'en', 'usa': 'en', 'uk': 'en', 'america': 'en', 'britain': 'en',
    
    # Brazilian Portuguese (separate from European Portuguese)
    'pt-br': 'pt-br', 'ptbr': 'pt-br', 'brazilian': 'pt-br', 'brasilian': 'pt-br',
    'brazil': 'pt-br', 'brasil': 'pt-br', 'br': 'pt-br', 'português brasileiro': 'pt-br',
    
    # European Portuguese (separate from Brazilian)
    'pt-pt': 'pt-pt', 'portugal': 'pt-pt', 'european portuguese': 'pt-pt',
    'portuguese': 'pt-pt', 'português': 'pt-pt', 'portugues': 'pt-pt',
    
    # French variations
    'fr': 'fr', 'french': 'fr', 'français': 'fr', 'francais': 'fr', 
    'france': 'fr', 'français de france': 'fr',
    
    # Spanish variations
    'es': 'es', 'spanish': 'es', 'español': 'es', 'espanol': 'es',
    'spain': 'es', 'españa': 'es', 'castilian': 'es',
    
    # Italian variations
    'it': 'it', 'italian': 'it', 'italiano': 'it', 'italy': 'it', 'italia': 'it',
    
    # Norwegian variations
    'no': 'no', 'norwegian': 'no', 'norsk': 'no', 'norway': 'no', 'norge': 'no',
    
    # Dutch variations
    'nl': 'nl', 'dutch': 'nl', 'nederlands': 'nl', 'netherlands': 'nl', 'holland': 'nl',
    
    # Japanese variations
    'ja': 'ja', 'japanese': 'ja', '日本語': 'ja', 'japan': 'ja', 'nihongo': 'ja',
    
    # Chinese variations
    'zh': 'zh', 'chinese': 'zh', '中文': 'zh', 'china': 'zh',
    'zh-cn': 'zh-cn', 'mandarin': 'zh-cn', 'simplified': 'zh-cn', 'mainland': 'zh-cn',
    'zh-tw': 'zh-tw', 'traditional': 'zh-tw', 'taiwan': 'zh-tw', 'taiwanese': 'zh-tw',
    
    # Russian variations
    'ru': 'ru', 'russian': 'ru', 'русский': 'ru', 'russia': 'ru', 'россия': 'ru',
    
    # Korean variations
    'ko': 'ko', 'korean': 'ko', '한국어': 'ko', 'korea': 'ko', 'south korea': 'ko',
    
    # Indian Languages (F5-Hindi-Small for Hindi, others use base F5TTS models)
    
    # Hindi variations
    'hi': 'hi', 'hindi': 'hi', 'हिन्दी': 'hi', 'hin': 'hi', 'देवनागरी': 'hi',
    
    # Assamese variations
    'as': 'as', 'assamese': 'as', 'অসমীয়া': 'as', 'asom': 'as', 'axomiya': 'as',
    
    # Bengali variations  
    'bn': 'bn', 'bengali': 'bn', 'বাংলা': 'bn', 'bangla': 'bn', 'west bengal': 'bn',
    'bangladesh': 'bn', 'bengal': 'bn',
    
    # Gujarati variations
    'gu': 'gu', 'gujarati': 'gu', 'ગુજરાતી': 'gu', 'gujarat': 'gu', 'gujrati': 'gu',
    
    # Kannada variations
    'kn': 'kn', 'kannada': 'kn', 'ಕನ್ನಡ': 'kn', 'karnataka': 'kn', 'kanarese': 'kn',
    
    # Malayalam variations
    'ml': 'ml', 'malayalam': 'ml', 'മലയാളം': 'ml', 'kerala': 'ml', 'malayali': 'ml',
    
    # Marathi variations
    'mr': 'mr', 'marathi': 'mr', 'मराठी': 'mr', 'maharashtra': 'mr',
    
    # Odia variations
    'or': 'or', 'odia': 'or', 'ଓଡ଼ିଆ': 'or', 'oriya': 'or', 'odisha': 'or', 'orissa': 'or',
    
    # Punjabi variations
    'pa': 'pa', 'punjabi': 'pa', 'ਪੰਜਾਬੀ': 'pa', 'panjabi': 'pa', 'punjab': 'pa',
    
    # Tamil variations
    'ta': 'ta', 'tamil': 'ta', 'தமிழ்': 'ta', 'tamil nadu': 'ta', 'tamilnadu': 'ta',
    
    # Telugu variations
    'te': 'te', 'telugu': 'te', 'తెలుగు': 'te', 'andhra pradesh': 'te',
    'andhra': 'te', 'telangana': 'te',
    
    # === ChatterBox Official 23-Lang Additional Languages ===
    
    # Arabic variations
    'ar': 'ar', 'arabic': 'ar', 'العربية': 'ar', 'arab': 'ar', 'middle east': 'ar',
    
    # Danish variations  
    'da': 'da', 'danish': 'da', 'dansk': 'da', 'denmark': 'da', 'danmark': 'da',
    
    # Greek variations
    'el': 'el', 'greek': 'el', 'ελληνικά': 'el', 'greece': 'el', 'hellenic': 'el',
    'gr': 'el',  # Common abbreviation for Greece -> Greek language
    
    # Finnish variations
    'fi': 'fi', 'finnish': 'fi', 'suomi': 'fi', 'finland': 'fi', 'suomalainen': 'fi',
    
    # Hebrew variations
    'he': 'he', 'hebrew': 'he', 'עברית': 'he', 'israel': 'he', 'israeli': 'he',
    'iw': 'he',  # Legacy ISO code
    
    # Malay variations
    'ms': 'ms', 'malay': 'ms', 'bahasa melayu': 'ms', 'malaysia': 'ms', 'melayu': 'ms',
    
    # Polish variations
    'pl': 'pl', 'polish': 'pl', 'polski': 'pl', 'poland': 'pl', 'polska': 'pl',
    
    # Swedish variations
    'sv': 'sv', 'swedish': 'sv', 'svenska': 'sv', 'sweden': 'sv', 'sverige': 'sv',
    
    # Swahili variations
    'sw': 'sw', 'swahili': 'sw', 'kiswahili': 'sw', 'tanzania': 'sw', 'kenya': 'sw',
    
    # Turkish variations
    'tr': 'tr', 'turkish': 'tr', 'türkçe': 'tr', 'turkce': 'tr', 'turkey': 'tr',
    'türkiye': 'tr', 'turkiye': 'tr',
    
    # Additional European languages not in character parser
    # Czech variations
    'cs': 'cs', 'cz': 'cs', 'czech': 'cs', 'čeština': 'cs', 'ceska': 'cs',
    
    # Slovak variations 
    'sk': 'sk', 'slovak': 'sk', 'slovenčina': 'sk', 'slovakia': 'sk',
    
    # Hungarian variations
    'hu': 'hu', 'hungarian': 'hu', 'magyar': 'hu', 'hungary': 'hu',
    
    # Romanian variations
    'ro': 'ro', 'romanian': 'ro', 'română': 'ro', 'romania': 'ro',
    
    # Bulgarian variations
    'bg': 'bg', 'bulgarian': 'bg', 'български': 'bg', 'bulgaria': 'bg',
    
    # Croatian variations
    'hr': 'hr', 'croatian': 'hr', 'hrvatski': 'hr', 'croatia': 'hr',
    
    # Serbian variations
    'sr': 'sr', 'serbian': 'sr', 'српски': 'sr', 'serbia': 'sr',
    
    # Slovenian variations
    'sl': 'sl', 'slovenian': 'sl', 'slovenščina': 'sl', 'slovenia': 'sl',
    
    # Estonian variations
    'et': 'et', 'estonian': 'et', 'eesti': 'et', 'estonia': 'et',
    
    # Latvian variations
    'lv': 'lv', 'latvian': 'lv', 'latviešu': 'lv', 'latvia': 'lv',
    
    # Lithuanian variations
    'lt': 'lt', 'lithuanian': 'lt', 'lietuvių': 'lt', 'lithuania': 'lt',
    
    # Icelandic variations
    'is': 'is', 'icelandic': 'is', 'íslenska': 'is', 'iceland': 'is',
    
    # Additional Asian/African languages
    # Vietnamese variations
    'vi': 'vi', 'vietnamese': 'vi', 'tiếng việt': 'vi', 'vietnam': 'vi',
    
    # Indonesian variations
    'id': 'id', 'indonesian': 'id', 'bahasa indonesia': 'id', 'indonesia': 'id',
    
    # Filipino/Tagalog variations
    'tl': 'tl', 'fil': 'tl', 'filipino': 'tl', 'tagalog': 'tl', 'philippines': 'tl',
    
    # Persian/Farsi variations
    'fa': 'fa', 'persian': 'fa', 'farsi': 'fa', 'فارسی': 'fa', 'iran': 'fa',
    
    # Urdu variations
    'ur': 'ur', 'urdu': 'ur', 'اردو': 'ur', 'pakistan': 'ur',
    
    # Afrikaans variations
    'af': 'af', 'afrikaans': 'af', 'south africa': 'af',
    
    # Zulu variations
    'zu': 'zu', 'zulu': 'zu', 'isizulu': 'zu',
    
    # Additional common abbreviations and alternatives
    'jp': 'ja',  # Common abbreviation for Japanese
    'kr': 'ko',  # Common abbreviation for Korean
    'se': 'sv',  # Common abbreviation for Swedish
    'dk': 'da',  # Common abbreviation for Danish
}


def resolve_language_alias(language_input: str) -> str:
    """
    Resolve language alias to canonical language code.
    
    Args:
        language_input: User input language (e.g., "German", "brasil", "pt-BR")
        
    Returns:
        Canonical language code (e.g., "de", "pt-br")
    """
    # Normalize input: lowercase and strip whitespace
    normalized = language_input.strip().lower()
    
    # Look up in aliases
    canonical = LANGUAGE_ALIASES.get(normalized)
    if canonical:
        return canonical
        
    # If no alias found, return the original (for backward compatibility)
    return normalized


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
            elif self.engine_type == 'index_tts':
                # IndexTTS-2 is multilingual and uses the same model for all languages
                return default_model or 'IndexTTS-2'  # Use configured model
        
        # Check if language is supported
        if lang_code in engine_mappings:
            return engine_mappings[lang_code]
        else:
            # Handle IndexTTS-2 specifically - it's multilingual so doesn't need language-specific models
            if self.engine_type == 'index_tts':
                return default_model or 'IndexTTS-2'  # Use same model for all languages

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
                "ja": "F5-JP",       # Japanese (alternative code)
                "th": "F5-TH",       # Thai
                "pt": "F5-PT-BR",    # Portuguese (Brazil)
                "pt-br": "F5-PT-BR", # Portuguese (Brazil) - alternative format
                "pl": "F5-Polish",   # Polish - high quality model from Gregniuki
                "hi": "F5-Hindi-Small",  # Hindi - uses Small model from IIT Madras
                # Note: Other languages fall back to default_model and use phonemization when appropriate
            },
            "chatterbox": chatterbox_mappings,
            "vibevoice": {
                "en": "vibevoice-1.5B",  # This will be overridden by default_model
                "zh": "vibevoice-1.5B",  # Chinese - same model supports both EN/ZH
                "zh-cn": "vibevoice-1.5B",  # Simplified Chinese
                "chinese": "vibevoice-1.5B",  # Alternative format
                # VibeVoice models support both English and Chinese with the same model
            },
            "index_tts": {
                # IndexTTS-2 is multilingual - same model handles all languages
                # No specific mappings needed as it uses character parser for language detection
                # All languages fallback to the configured model
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
index_tts_language_mapper = LanguageModelMapper("index_tts")


def get_language_mapper(engine_type: str) -> LanguageModelMapper:
    """
    Get language mapper instance for specified engine.

    Args:
        engine_type: "f5tts", "chatterbox", "vibevoice", or "index_tts"

    Returns:
        LanguageModelMapper instance
    """
    if engine_type == "f5tts":
        return f5tts_language_mapper
    elif engine_type == "chatterbox":
        return chatterbox_language_mapper
    elif engine_type == "vibevoice":
        return vibevoice_language_mapper
    elif engine_type == "index_tts":
        return index_tts_language_mapper
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


def get_model_for_language(engine_type: str, lang_code: str, default_model: str) -> str:
    """
    Convenience function to get model for language.

    Args:
        engine_type: "f5tts", "chatterbox", "vibevoice", or "index_tts"
        lang_code: Language code
        default_model: Default model for base language

    Returns:
        Model name for the specified language
    """
    mapper = get_language_mapper(engine_type)
    return mapper.get_model_for_language(lang_code, default_model)