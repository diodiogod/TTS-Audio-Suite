"""
Test script for modular multilingual architecture
Simple test to verify the modular components work together
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

def test_language_mapper():
    """Test language model mapper functionality."""
    print("🧪 Testing Language Model Mapper...")
    
    from core.language_model_mapper import get_language_mapper, get_model_for_language
    
    # Test F5-TTS mapper
    f5tts_mapper = get_language_mapper("f5tts")
    assert f5tts_mapper.get_model_for_language("en", "F5TTS_v1_Base") == "F5TTS_v1_Base"
    assert f5tts_mapper.get_model_for_language("de", "F5TTS_v1_Base") == "F5-DE"
    assert f5tts_mapper.get_model_for_language("fr", "F5TTS_v1_Base") == "F5-FR"
    
    # Test ChatterBox mapper
    chatterbox_mapper = get_language_mapper("chatterbox")
    assert chatterbox_mapper.get_model_for_language("en", "English") == "English"
    assert chatterbox_mapper.get_model_for_language("de", "English") == "German"
    assert chatterbox_mapper.get_model_for_language("no", "English") == "Norwegian"
    
    # Test convenience function
    assert get_model_for_language("f5tts", "es", "F5TTS_Base") == "F5-ES"
    assert get_model_for_language("chatterbox", "es", "English") == "Spanish"
    
    print("✅ Language Model Mapper tests passed!")


def test_multilingual_engine_basic():
    """Test basic multilingual engine functionality."""
    print("🧪 Testing Multilingual Engine (basic functionality)...")
    
    from core.multilingual_engine import MultilingualEngine
    
    # Create engine instances
    f5tts_engine = MultilingualEngine("f5tts")
    chatterbox_engine = MultilingualEngine("chatterbox")
    
    # Test basic properties
    assert f5tts_engine.engine_type == "f5tts"
    assert f5tts_engine.sample_rate == 24000
    
    assert chatterbox_engine.engine_type == "chatterbox"
    assert chatterbox_engine.sample_rate == 44100
    
    # Test multilingual detection
    simple_text = "Hello world"
    multilingual_text = "Hello [de:Alice] Hallo [fr:] Bonjour"
    
    assert not f5tts_engine.is_multilingual_or_multicharacter(simple_text)
    assert f5tts_engine.is_multilingual_or_multicharacter(multilingual_text)
    
    print("✅ Multilingual Engine basic tests passed!")


def test_character_parser_integration():
    """Test character parser integration with modular system."""
    print("🧪 Testing Character Parser Integration...")
    
    from core.character_parser import character_parser
    
    # Test the fixed [fr:] parsing
    test_text = "[no:Bob] Hei! [fr:] Bonjour!"
    segments = character_parser.split_by_character_with_language(test_text)
    
    print(f"📝 Parsed segments: {segments}")
    
    # Should have 2 segments
    assert len(segments) == 2
    
    # First segment: Bob in Norwegian
    assert segments[0][0] == "male_01"  # Bob maps to male_01
    assert segments[0][1] == "Hei!"
    assert segments[0][2] == "no"
    
    # Second segment: narrator in French 
    assert segments[1][0] == "narrator"  # [fr:] should default to narrator
    assert segments[1][1] == "Bonjour!"
    assert segments[1][2] == "fr"
    
    print("✅ Character Parser Integration tests passed!")


def main():
    """Run all modular architecture tests."""
    print("🚀 Testing Modular Multilingual Architecture")
    print("=" * 50)
    
    try:
        test_language_mapper()
        test_multilingual_engine_basic() 
        test_character_parser_integration()
        
        print("=" * 50)
        print("🎉 All modular architecture tests passed!")
        print("✅ Ready to integrate with nodes!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)