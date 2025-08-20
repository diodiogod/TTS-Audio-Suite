"""
Higgs Audio Engine Node - Higgs Audio-specific configuration for TTS Audio Suite
Provides Higgs Audio engine adapter with all Higgs Audio-specific parameters
"""

import os
import sys
import importlib.util
from typing import Dict, Any

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


class HiggsAudioEngineNode(BaseTTSNode):
    """
    Higgs Audio Engine configuration node.
    Provides Higgs Audio-specific parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ Higgs Audio 2 Engine"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Import Higgs Audio models for dropdown
        try:
            from engines.higgs_audio.higgs_audio_downloader import HIGGS_AUDIO_MODELS
            available_models = list(HIGGS_AUDIO_MODELS.keys())
            
            # Add local models
            from engines.higgs_audio.higgs_audio import HiggsAudioEngine
            engine = HiggsAudioEngine()
            all_models = engine.get_available_models()
            
            # Combine and deduplicate
            available_models.extend([m for m in all_models if m not in available_models])
        except ImportError:
            available_models = ["higgs-audio-v2-3B"]
        
        # Load voice presets
        try:
            from engines.higgs_audio.higgs_audio import HiggsAudioEngine
            engine = HiggsAudioEngine()
            voice_presets, _ = engine.load_voice_presets()
            voice_preset_list = list(voice_presets.keys())
        except:
            voice_preset_list = ["voice_clone", "en_woman", "en_man", "belinda"]
        
        return {
            "required": {
                "model": (available_models, {
                    "default": "higgs-audio-v2-3B",
                    "tooltip": "Higgs Audio 2 model selection:\n• higgs-audio-v2-3B: Main 3B parameter model with best quality and voice cloning capabilities\n• Future models will appear here when available\n\nThe model handles voice cloning, multi-speaker generation, and natural speech synthesis."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Computation device selection:\n• auto: Automatically choose CUDA GPU if available, fallback to CPU\n• cuda: Force GPU acceleration (requires NVIDIA GPU with CUDA)\n• cpu: Force CPU-only processing (slower but works on any hardware)\n\nRecommended: Leave on 'auto' unless you have specific hardware requirements."
                }),
                "multi_speaker_mode": (["Custom Character Switching", "Native Multi-Speaker (Conversation)", "Native Multi-Speaker (System Context)"], {
                    "default": "Custom Character Switching",
                    "tooltip": "IMPORTANT: Each mode requires different text formats!\n\n• Custom Character Switching: ⭐ MAIN METHOD - Use ANY character names like [Alice], [Bob], [Narrator]. Each segment generated separately with character-specific voice files from voices folder. Supports [pause:2] tags. Most flexible and reliable.\n\n• Native Multi-Speaker (Conversation): Higgs Audio 2's native mode. MUST use [SPEAKER0] and [SPEAKER1] tags only! Requires opt_second_narrator input. NO pause tag support.\n\n• Native Multi-Speaker (System Context): ⚠️ EXPERIMENTAL - Higgs Audio 2's native mode. MUST use [SPEAKER0] and [SPEAKER1] tags only! May produce audio artifacts. NO pause tag support."
                }),
                "voice_preset": (voice_preset_list, {
                    "default": "voice_clone",
                    "tooltip": "Voice style selection:\n• voice_clone: 🎯 RECOMMENDED - Clone the connected reference voice (from Character Voices or TTS Text narrator input)\n• en_woman/en_man: Built-in English voices (ignores reference audio)\n• belinda, chadwick, vex: Built-in character voices with distinct personalities\n\nFor best results, use 'voice_clone' with high-quality reference audio from Character Voices node."
                }),
                "system_prompt": ("STRING", {
                    "default": "Generate audio following instruction.",
                    "multiline": True,
                    "tooltip": "System instruction that guides how Higgs Audio 2 generates speech:\n\n• Default: 'Generate audio following instruction.' - Works for most cases\n• Custom examples:\n  - 'Speak clearly and slowly.' - For clearer pronunciation\n  - 'Generate dramatic, emotional speech.' - For expressive delivery\n  - 'Speak in a calm, professional tone.' - For business/formal content\n\nThis is an advanced parameter - the default usually works best unless you need specific speech characteristics."
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "🌡️ Controls speech creativity and randomness:\n\n• 0.0-0.5: Very predictable, robotic speech (not recommended)\n• 0.6-0.8: 🎯 Conservative, natural speech with good consistency\n• 1.0: 🎯 RECOMMENDED - Balanced natural variation\n• 1.2-1.5: More expressive, varied pronunciation and pacing\n• 1.8-2.0: Highly creative but potentially unstable\n\nStart with 1.0. Lower for consistent corporate voices, raise for dramatic/character voices."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "🎯 Nucleus sampling - controls vocabulary diversity:\n\n• 0.1-0.3: Very limited vocabulary, may sound repetitive\n• 0.7-0.85: 🎯 Conservative, clear pronunciation\n• 0.9-0.95: 🎯 RECOMMENDED - Good balance of clarity and natural variation\n• 0.98-1.0: Maximum vocabulary diversity, may include rare pronunciations\n\nHigher values = more varied speech patterns. 0.95 works well for most content."
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "🔢 Limits vocabulary choices per word:\n\n• -1: Disabled (uses only top_p)\n• 10-30: 🎯 Very focused, consistent pronunciation\n• 40-60: 🎯 RECOMMENDED - Good balance of consistency and natural variation\n• 80-100: Maximum vocabulary freedom, more diverse but potentially inconsistent\n\nWorks together with top_p. Lower values = more predictable speech. 50 is ideal for most use cases."
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 128,
                    "max": 4096,
                    "step": 128,
                    "tooltip": "🔤 Maximum AI tokens per text chunk - affects processing and quality:\n\n• 512-1024: 🎯 Short texts, faster processing, good for simple sentences\n• 1536-2048: 🎯 RECOMMENDED - Balanced processing speed and quality for most content\n• 2560-4096: Long texts, slower but better context understanding for complex passages\n\nHigher values allow longer coherent speech but increase processing time. 2048 handles most content well."
                })
            },
            "optional": {
                "opt_second_narrator": ("AUDIO", {
                    "tooltip": "Second narrator voice for native multi-speaker modes. Used as SPEAKER1 voice when multi_speaker_mode is set to Native Multi-Speaker. Only needed for native modes, ignored in Custom Character Switching mode. First narrator (from Character Voices or TTS Text) becomes SPEAKER0."
                })
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("tts_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "🎤 TTS Audio Suite/Engines"
    DESCRIPTION = "Configure Higgs Audio 2 engine for TTS generation with voice cloning"
    
    def create_engine_config(self, model, device, multi_speaker_mode, voice_preset, system_prompt,
                           temperature, top_p, top_k, max_new_tokens, opt_second_narrator=None):
        """Create Higgs Audio engine configuration"""
        
        # Validate parameters
        config = {
            "engine_type": "higgs_audio",
            "model": model,
            "device": device,
            "multi_speaker_mode": multi_speaker_mode,
            "voice_preset": voice_preset,
            "system_prompt": system_prompt,
            "temperature": max(0.0, min(2.0, temperature)),
            "top_p": max(0.1, min(1.0, top_p)),
            "top_k": max(-1, min(100, top_k)),
            "max_new_tokens": max(128, min(4096, max_new_tokens)),
            "opt_second_narrator": opt_second_narrator,
            "adapter_class": "HiggsAudioEngineAdapter"
        }
        
        print(f"✅ Higgs Audio engine config created: {model} on {device}")
        return (config,)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "HiggsAudioEngineNode": HiggsAudioEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiggsAudioEngineNode": "⚙️ Higgs Audio 2 Engine"
}