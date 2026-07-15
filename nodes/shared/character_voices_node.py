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

from utils.voice.discovery import get_available_characters, get_available_voices, load_voice_reference
from utils.voice.character_saver import read_character_metadata


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
        # INPUT_TYPES can be queried repeatedly by ComfyUI. Use the discovery
        # cache here; explicit rescans belong to the Refresh Voice Cache node.
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

Selecting a library voice loads its transcription here automatically. Edits are temporary workflow overrides and never modify the source .txt file."""
                }),
                "trim_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "Start of the derived reference clip in seconds. The custom timeline controls this value."
                }),
                "trim_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100000.0,
                    "step": 0.01,
                    "tooltip": "End of the derived reference clip in seconds. 0 means the end of the source audio."
                }),
                "customized": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Internal UI state: the selected library voice has a temporary text or trim override."
                }),
            },
            "optional": {
                "opt_audio_input": ("AUDIO", {
                    "tooltip": "Direct audio input for voice reference (used when voice_name is 'none' or to override selected voice)"
                }),
            }
        }

    RETURN_TYPES = ("NARRATOR_VOICE", "STRING", "AUDIO")
    RETURN_NAMES = ("opt_narrator", "character_name", "reference_audio_only")
    FUNCTION = "get_voice_reference"
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"

    @staticmethod
    def _trim_audio(audio_tensor, trim_start: float, trim_end: float):
        """Return the effective audio and whether a non-full trim was applied."""
        if not isinstance(audio_tensor, dict):
            return audio_tensor, False

        waveform = audio_tensor.get("waveform")
        sample_rate = int(audio_tensor.get("sample_rate", 0) or 0)
        if waveform is None or sample_rate <= 0:
            return audio_tensor, False

        total_samples = int(waveform.shape[-1])
        duration = total_samples / sample_rate
        start = max(0.0, min(float(trim_start or 0.0), duration))
        requested_end = float(trim_end or 0.0)
        end = duration if requested_end <= 0.0 else max(0.0, min(requested_end, duration))

        trim_applied = start > 1e-6 or end < duration - 1e-6
        if not trim_applied:
            return audio_tensor, False
        if end <= start:
            raise ValueError(
                f"Invalid Character Voices trim range: start {start:.2f}s must be before end {end:.2f}s"
            )

        start_sample = min(total_samples, max(0, round(start * sample_rate)))
        end_sample = min(total_samples, max(start_sample + 1, round(end * sample_rate)))
        return {
            "waveform": waveform[..., start_sample:end_sample].contiguous(),
            "sample_rate": sample_rate,
        }, True

    @staticmethod
    def _format_audio_output(audio_tensor):
        """Format the effective reference as a standard ComfyUI AUDIO output."""
        if not isinstance(audio_tensor, dict) or "waveform" not in audio_tensor:
            return None

        waveform = audio_tensor["waveform"]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        return {
            "waveform": waveform.cpu().float(),
            "sample_rate": int(audio_tensor["sample_rate"]),
        }

    def get_voice_reference(
        self,
        voice_name: str,
        reference_text: str,
        opt_audio_input=None,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
        customized: bool = False,
    ):
        """
        Get voice reference for TTS engines.
        
        Args:
            voice_name: Selected voice from dropdown
            reference_text: Text reference for voice cloning
            opt_audio_input: Optional direct audio input
            trim_start: Start of the effective reference range in seconds
            trim_end: End of the effective reference range, or 0 for source end
            customized: Whether the UI transcription is a derived override
            
        Returns:
            Tuple of (narrator_voice_data, character_name, effective_audio)
        """
        try:
            used_folder_text = False
            voice_source = None
            source_audio_path = None
            canonical_reference_text = ""
            # Determine audio source and character name
            if opt_audio_input is not None:
                # Use direct audio input
                audio_path = None
                audio_tensor = opt_audio_input
                character_name = "direct_input"
                voice_source = "direct"
                customized = True
                print("🎭 Character Voices: Using direct audio input")
            elif voice_name != "none":
                # Load from voice folder
                audio_path, folder_reference_text = load_voice_reference(voice_name)
                source_audio_path = audio_path
                canonical_reference_text = folder_reference_text or ""
                
                if audio_path and os.path.exists(audio_path):
                    # Load audio tensor when possible, but do not fail the node if local
                    # decoding is unavailable. Several engines can work directly from the
                    # file path, and the browser preview already proved the file itself is
                    # valid. Hard-failing here breaks restored workflows for no reason.
                    audio_tensor = None
                    try:
                        from utils.audio.processing import AudioProcessingUtils
                        waveform, sample_rate = AudioProcessingUtils.safe_load_audio(audio_path)

                        # Audio is automatically normalized by safe_load_audio() to [-1, 1] range
                        # Convert to mono if stereo
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        audio_tensor = {"waveform": waveform, "sample_rate": sample_rate}
                    except Exception as e:
                        print(f"⚠️ Character Voices: Failed to decode audio tensor, using file path only: {audio_path} ({e})")

                    character_name = os.path.splitext(os.path.basename(voice_name))[0]
                    voice_source = "folder"

                    # Restored legacy workflows keep canonical disk text. New frontend
                    # edits explicitly mark the selected library voice as customized.
                    if not customized or not (reference_text and reference_text.strip()):
                        reference_text = canonical_reference_text
                        used_folder_text = True
                else:
                    print(f"⚠️ Character Voices: Voice file not found: {voice_name}")
                    return None, "", None
                    
            else:
                # No voice specified
                print("⚠️ Character Voices: No voice specified - provide voice_name or opt_audio_input")
                return None, "", None

            audio_tensor, trim_applied = self._trim_audio(audio_tensor, trim_start, trim_end)
            customized = bool(customized or trim_applied or voice_source == "direct")

            # A customized library voice must never expose the original untrimmed path
            # as effective audio. Keep it only as provenance for the UI and diagnostics.
            effective_audio_path = None if customized and audio_tensor is not None else audio_path

            # Create narrator voice data structure
            narrator_voice_data = {
                "audio": audio_tensor,
                "audio_path": effective_audio_path,
                "source_audio_path": source_audio_path,
                "reference_text": reference_text.strip() if reference_text else "",
                "canonical_reference_text": canonical_reference_text,
                "character_name": character_name,
                "source": voice_source,
                "customized": customized,
                "trim_start": float(trim_start or 0.0),
                "trim_end": float(trim_end or 0.0),
            }
            if source_audio_path:
                # Restore optional designer provenance without changing the
                # Character Voices node's own source/customization semantics.
                narrator_voice_data.update(read_character_metadata(source_audio_path))

            # Add validation info
            has_audio = audio_tensor is not None or bool(narrator_voice_data.get("audio_path"))
            has_text = bool(reference_text and reference_text.strip())
            
            if has_audio and has_text:
                compatibility = "F5-TTS, ChatterBox, and future engines"
            elif has_audio and not has_text:
                compatibility = "ChatterBox and audio-only engines"
            else:
                compatibility = "Limited compatibility - missing audio"
            
            ref_source = f" (text from {voice_name})" if used_folder_text else ""
            print(f"💬 Narrator Voice: {character_name} ready for {compatibility}{ref_source}")
            
            return narrator_voice_data, character_name, self._format_audio_output(audio_tensor)
            
        except Exception as e:
            print(f"❌ Character Voices error: {e}")
            import traceback
            traceback.print_exc()
            return None, "", None


# Register the node class
NODE_CLASS_MAPPINGS = {
    "CharacterVoicesNode": CharacterVoicesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterVoicesNode": "🎭 Character Voices"
}
