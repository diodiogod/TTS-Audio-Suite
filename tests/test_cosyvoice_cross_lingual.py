
import os
import sys
import torch
import torchaudio
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add ComfyUI root to path for folder_paths module
# sys.path.append(r"C:\_stability_matrix\Data\Packages\Comfy-new")
# Instead of adding path, we MOCK folder_paths because it might have complex dependencies
import sys
from unittest.mock import MagicMock
if 'folder_paths' not in sys.modules:
    mock_fp = MagicMock()
    mock_fp.get_folder_paths.return_value = []
    sys.modules['folder_paths'] = mock_fp

from engines.cosyvoice.cosyvoice import CosyVoiceEngine

# Subclass to bypass folder_paths dependency
class TestEngine(CosyVoiceEngine):
    def _find_model_directory(self, model_identifier):
        # Return absolute path directly
        return r"C:\_stability_matrix\Data\Packages\Comfy-new\models\TTS\CosyVoice\Fun-CosyVoice3-0.5B"

def test_cross_lingual_english():
    # Setup paths
    base_path = r"C:\_stability_matrix\Data\Packages\Comfy-new\custom_nodes\TTS-Audio-Suite"
    model_path = r"C:\_stability_matrix\Data\Packages\Comfy-new\models\TTS\CosyVoice\Fun-CosyVoice3-0.5B"
    output_dir = os.path.join(base_path, "tests", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prompt WAV (David Attenborough)
    prompt_wav = os.path.join(base_path, "voices_examples", "David_Attenborough CC3.wav")
    
    # Text to synthesize (English) with <|en|> tag as per official example
    text = "<|en|>The quick brown fox jumps over the lazy dog. This is a test of the cross lingual mode."
    
    print(f"Testing Cross Lingual Mode with text: {text}")
    print(f"Model path: {model_path}")
    
    try:
        # Initialize Engine with correct signature
        # We pass use_fp16=True, others default
        engine = TestEngine(
            model_dir="Fun-CosyVoice3-0.5B", # Dummy value, ignored by override
            use_fp16=True
        )
        
        # Call generate_cross_lingual

        # Note: cross_lingual does not use prompt_text (transcript)
        audio = engine.generate_cross_lingual(
            text=text,
            prompt_wav=prompt_wav,
            speed=1.0,
            text_frontend=False # We try False as well as checking if detection works
        )
        
        # Save output
        output_path = os.path.join(output_dir, "test_cross_lingual_english.wav")
        
        # Determine sample rate (should be 24000)
        sample_rate = engine.get_sample_rate()
        print(f"Output shape: {audio.shape}, Sample Rate: {sample_rate}")
        
        torchaudio.save(output_path, audio, sample_rate)
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cross_lingual_english()
