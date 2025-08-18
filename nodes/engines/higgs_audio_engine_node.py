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
                    "tooltip": "Higgs Audio model to use for generation"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run the model on"
                }),
                "voice_preset": (voice_preset_list, {
                    "default": "voice_clone",
                    "tooltip": "Voice preset to use for generation (voice_clone = use reference audio)"
                }),
                "audio_priority": (["auto", "preset_dropdown", "reference_input", "force_preset"], {
                    "default": "auto",
                    "tooltip": "Which audio source takes priority for voice cloning"
                }),
                "system_prompt": ("STRING", {
                    "default": "Generate audio following instruction.",
                    "multiline": True,
                    "tooltip": "System prompt to guide audio generation behavior"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature (0.8 = more stable, 1.2 = more varied)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling parameter (affects pronunciation variation)"
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-k sampling parameter (-1 = disabled, 50 = default)"
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 128,
                    "max": 4096,
                    "step": 128,
                    "tooltip": "Maximum tokens to generate per chunk (affects audio length and pacing)"
                })
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("tts_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "🎤 TTS Audio Suite/Engines"
    DESCRIPTION = "Configure Higgs Audio 2 engine for TTS generation with voice cloning"
    
    def create_engine_config(self, model, device, voice_preset, audio_priority, system_prompt,
                           temperature, top_p, top_k, max_new_tokens):
        """Create Higgs Audio engine configuration"""
        
        # Validate parameters
        config = {
            "engine_type": "higgs_audio",
            "model": model,
            "device": device,
            "voice_preset": voice_preset,
            "audio_priority": audio_priority,
            "system_prompt": system_prompt,
            "temperature": max(0.0, min(2.0, temperature)),
            "top_p": max(0.1, min(1.0, top_p)),
            "top_k": max(-1, min(100, top_k)),
            "max_new_tokens": max(128, min(4096, max_new_tokens)),
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