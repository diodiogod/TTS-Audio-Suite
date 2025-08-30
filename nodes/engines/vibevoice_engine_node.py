"""
VibeVoice Engine Node - VibeVoice-specific configuration for TTS Audio Suite
Provides VibeVoice engine adapter with all VibeVoice-specific parameters
"""

import os
import sys
import importlib.util
from typing import Dict, Any

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

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


class VibeVoiceEngineNode(BaseTTSNode):
    """
    VibeVoice Engine configuration node.
    Provides VibeVoice-specific parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ VibeVoice Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Import VibeVoice models for dropdown
        try:
            from engines.vibevoice_engine.vibevoice_downloader import VIBEVOICE_MODELS
            from engines.vibevoice_engine.vibevoice_engine import VibeVoiceEngine
            
            engine = VibeVoiceEngine()
            available_models = engine.get_available_models()
        except ImportError:
            available_models = ["vibevoice-1.5B", "vibevoice-7B"]
        
        return {
            "required": {
                "model": (available_models, {
                    "default": "vibevoice-1.5B",
                    "tooltip": "VibeVoice model selection:\n• vibevoice-1.5B: Official Microsoft model (2.7B params) - Faster, 90-min generation\n• vibevoice-7B: Community preview (9.3B params) - Better quality, 45-min generation\n\nBoth support multi-speaker and long-form generation."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Computation device:\n• auto: Automatically choose best available\n• cuda: Force GPU (requires NVIDIA GPU, ~7GB VRAM)\n• cpu: Force CPU processing (slower)\n\nRecommended: 'auto' for automatic selection."
                }),
                "multi_speaker_mode": (["Custom Character Switching", "Native Multi-Speaker"], {
                    "default": "Custom Character Switching",
                    "tooltip": "Speaker generation mode - SUPPORTS BOTH FORMATS!\n\n• Custom Character Switching: ⭐ RECOMMENDED - Use [Alice], [Bob] character tags. Each character generated separately with voice files from voices folder. Supports pause tags, per-character control, unlimited characters.\n\n• Native Multi-Speaker: ✅ TWO FORMAT OPTIONS:\n  1. [Alice], [Bob] tags → auto-converted to Speaker format\n  2. Manual 'Speaker 1: Hello\nSpeaker 2: Hi there' format\n\nUp to 4 speakers. More efficient single-pass generation.\n\n🔧 PRIORITY: Connected speaker2_voice/3/4 inputs override character aliases with warnings!"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.3,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Classifier-free guidance scale:\n• 1.0: Minimal guidance\n• 1.3: 🎯 RECOMMENDED - Balanced guidance\n• 1.5-2.0: Stronger text adherence\n\nHigher values follow text more closely but may sound less natural."
                }),
                "use_sampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Sampling mode:\n• False: 🎯 RECOMMENDED - Deterministic generation for consistency\n• True: Sampling with temperature/top_p for more variation\n\nDeterministic mode provides more reliable results."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "🌡️ Generation temperature (only with sampling):\n• 0.1-0.5: Very conservative\n• 0.8-1.0: 🎯 Natural variation\n• 1.2-2.0: More creative but less stable\n\nOnly used when use_sampling is True."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "🎯 Nucleus sampling (only with sampling):\n• 0.5-0.7: Focused vocabulary\n• 0.9-0.95: 🎯 RECOMMENDED - Balanced\n• 1.0: Full vocabulary\n\nOnly used when use_sampling is True."
                }),
                "chunk_minutes": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 90,
                    "step": 1,
                    "tooltip": "⏱️ Time-based chunking (OVERRIDES TTS Text chunking settings):\n• 0: Disabled - uses TTS Text node chunking settings\n• 5-10: Good for memory efficiency\n• 15-30: Balance between quality and memory\n\nWhen > 0, ignores TTS Text enable_chunking and max_chars_per_chunk. Converted to ~750 chars/min internally."
                }),
                "max_new_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 65536,
                    "step": 100,
                    "tooltip": "🔤 Maximum generation tokens:\n• 0: Auto (let model decide)\n• 1000-2000: Short content\n• 10000-20000: Medium content\n• 30000-65536: Long-form content\n\nSafety limit to prevent runaway generation. 0 recommended for auto."
                })
            },
            "optional": {
                "speaker2_voice": (any_typ, {
                    "tooltip": "🎤 Voice for Speaker 2 in Native Multi-Speaker mode. Connect audio input or Character Voices output. Used as second speaker when multi_speaker_mode is Native."
                }),
                "speaker3_voice": (any_typ, {
                    "tooltip": "🎤 Voice for Speaker 3 in Native Multi-Speaker mode. Connect audio input or Character Voices output. Up to 4 speakers supported."
                }),
                "speaker4_voice": (any_typ, {
                    "tooltip": "🎤 Voice for Speaker 4 in Native Multi-Speaker mode. Connect audio input or Character Voices output. Maximum of 4 speakers in native mode."
                })
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    DESCRIPTION = "Configure VibeVoice engine for multi-speaker TTS with 90-minute generation capability. Supports both custom character switching and native multi-speaker modes."
    
    def create_engine_config(self, model, device, multi_speaker_mode, cfg_scale,
                           use_sampling, temperature, top_p, chunk_minutes, max_new_tokens,
                           speaker2_voice=None, speaker3_voice=None, speaker4_voice=None):
        """Create VibeVoice engine configuration"""
        
        # Convert chunk_minutes to characters (approximately 750 chars per minute)
        # Based on: 150 words/min * 5 chars/word = 750 chars/min
        chunk_chars = chunk_minutes * 750 if chunk_minutes > 0 else 0
        
        # Validate parameters
        config = {
            "engine_type": "vibevoice",
            "model": model,
            "device": device,
            "multi_speaker_mode": multi_speaker_mode,
            "cfg_scale": max(1.0, min(2.0, cfg_scale)),
            "use_sampling": bool(use_sampling),
            "temperature": max(0.1, min(2.0, temperature)),
            "top_p": max(0.1, min(1.0, top_p)),
            "chunk_chars": chunk_chars,  # Backend uses characters
            "chunk_minutes": chunk_minutes,  # Store for UI reference
            "max_new_tokens": max_new_tokens if max_new_tokens > 0 else None,
            "speaker2_voice": speaker2_voice,
            "speaker3_voice": speaker3_voice,
            "speaker4_voice": speaker4_voice,
            "adapter_class": "VibeVoiceEngineAdapter"
        }
        
        # Display configuration
        print(f"🎙️ VibeVoice Engine configured:")
        print(f"   Model: {model} on {device}")
        print(f"   Mode: {multi_speaker_mode}")
        print(f"   CFG Scale: {cfg_scale}, Sampling: {use_sampling}")
        if chunk_minutes > 0:
            print(f"   Chunking: Every {chunk_minutes} minutes (~{chunk_chars} chars)")
        
        return (config,)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VibeVoiceEngineNode": VibeVoiceEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VibeVoiceEngineNode": "🎙️ VibeVoice Engine"
}