"""
Character Voices Node - Voice reference management for TTS Audio Suite
Provides unified voice reference handling for all TTS engines with audio/text support
"""

import torch
import os
import sys
import importlib.util

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from utils.voice.discovery import get_available_voices, load_voice_reference, get_available_characters


class CharacterVoicesNode(BaseTTSNode):
    """
    Character Voices Node - Unified voice reference management.
    Provides voice references for all TTS engines with flexible audio/text output.
    Replaces the opt_reference_text widget from F5-TTS nodes with centralized voice management.
    """
    
    @classmethod
    def NAME(cls):
        return "🎭 Character Voices"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available reference audio files from voice folders
        reference_files = get_available_voices()
        
        return {
            "required": {
                "voice_name": (reference_files, {
                    "default": "none",
                    "tooltip": """Use 'none' to rely on direct audio input + input text.

Select character voice from models/voices/ or voices_examples/ folders.

IMPORTANT: Character Voices node requires a .txt file with the same name as the audio file to recognize it as a character.

FILE REQUIREMENTS:
• filename.wav + filename.txt (basic setup)
• filename.wav + filename.reference.txt
• filename.wav + filename.txt + filename.reference.txt (both files)

PRIORITY SYSTEM - When both .txt and .reference.txt exist:
• .reference.txt = actual spoken text transcription (used for voice cloning)
• .txt = audio information/metadata (license, etc.)"""
                }),
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": """Create reference text on-the-fly for connected audio input.

ENGINE REQUIREMENTS:
• F5-TTS: REQUIRES reference text (must match spoken audio exactly)
• Higgs Audio 2: Optional but uses reference text if provided
• ChatterBox/VibeVoice/IndexTTS: Don't use reference text

Leave empty to use text from selected character's files."""
                }),
            },
            "optional": {
                "opt_audio_input": ("AUDIO", {
                    "tooltip": "Direct audio input for voice reference (used when voice_name is 'none' or to override selected voice)"
                }),
            }
        }

    RETURN_TYPES = ("NARRATOR_VOICE", "STRING")
    RETURN_NAMES = ("opt_narrator", "character_name")
    FUNCTION = "get_voice_reference"
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"

    def get_voice_reference(self, voice_name: str, reference_text: str, opt_audio_input=None):
        """
        Get voice reference for TTS engines.
        
        Args:
            voice_name: Selected voice from dropdown
            reference_text: Text reference for voice cloning
            opt_audio_input: Optional direct audio input
            
        Returns:
            Tuple of (narrator_voice_data, character_name)
        """
        try:
            used_folder_text = False
            # Determine audio source and character name
            if opt_audio_input is not None:
                # Use direct audio input
                audio_path = None
                audio_tensor = opt_audio_input
                character_name = "direct_input"
                print("🎭 Character Voices: Using direct audio input")
            elif voice_name != "none":
                # Load from voice folder
                audio_path, folder_reference_text = load_voice_reference(voice_name)
                
                if audio_path and os.path.exists(audio_path):
                    # Load audio tensor from file
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    audio_tensor = {"waveform": waveform, "sample_rate": sample_rate}
                    character_name = os.path.splitext(os.path.basename(voice_name))[0]
                    
                    # Use folder reference text if provided text is empty
                    if not reference_text.strip() and folder_reference_text:
                        reference_text = folder_reference_text
                        used_folder_text = True
                else:
                    print(f"⚠️ Character Voices: Voice file not found: {voice_name}")
                    return None, ""
                    
            else:
                # No voice specified
                print("⚠️ Character Voices: No voice specified - provide voice_name or opt_audio_input")
                return None, ""

            # Create narrator voice data structure
            narrator_voice_data = {
                "audio": audio_tensor,
                "audio_path": audio_path if 'audio_path' in locals() else None,
                "reference_text": reference_text.strip() if reference_text else "",
                "character_name": character_name,
                "source": "folder" if voice_name != "none" else "direct"
            }
            
            # Add validation info
            has_audio = audio_tensor is not None
            has_text = bool(reference_text.strip())
            
            if has_audio and has_text:
                compatibility = "F5-TTS, ChatterBox, and future engines"
            elif has_audio and not has_text:
                compatibility = "ChatterBox and audio-only engines"
            else:
                compatibility = "Limited compatibility - missing audio"
            
            ref_source = f" (text from {voice_name})" if used_folder_text else ""
            print(f"💬 Narrator Voice: {character_name} ready for {compatibility}{ref_source}")
            
            return narrator_voice_data, character_name
            
        except Exception as e:
            print(f"❌ Character Voices error: {e}")
            import traceback
            traceback.print_exc()
            return None, ""


# Register the node class
NODE_CLASS_MAPPINGS = {
    "CharacterVoicesNode": CharacterVoicesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterVoicesNode": "🎭 Character Voices"
}