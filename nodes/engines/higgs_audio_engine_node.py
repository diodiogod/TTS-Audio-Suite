"""
Higgs Audio Engine Node - Higgs Audio-specific configuration for TTS Audio Suite
Provides Higgs Audio engine adapter with all Higgs Audio-specific parameters
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


class HiggsAudioEngineNode(BaseTTSNode):
    """
    Higgs Audio Engine configuration node.
    Provides Higgs Audio-specific parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ Higgs Audio 2 Engine"
    
    @classmethod
    def _get_available_higgs_models(cls) -> list:
        """Get available Higgs Audio models without importing heavy modules.
        Reconstructs the discovery logic using file reading only."""
        # Static model definitions (not heavy)
        HIGGS_AUDIO_MODELS = {
            "higgs-audio-v2-3B": {
                "generation_repo": "bosonai/higgs-audio-v2-generation-3B-base",
                "tokenizer_repo": "bosonai/higgs-audio-v2-tokenizer",
            }
        }

        available = list(HIGGS_AUDIO_MODELS.keys())
        found_local_models = set()

        try:
            from utils.models.extra_paths import get_all_tts_model_paths

            # Search in all configured TTS paths (respects extra_model_paths.yaml)
            for base_tts_path in get_all_tts_model_paths('TTS'):
                # Try case variations for folder name
                for higgs_folder_name in ["HiggsAudio", "higgs_audio", "higgsaudio"]:
                    higgs_base_dir = os.path.join(base_tts_path, higgs_folder_name)
                    if not os.path.exists(higgs_base_dir):
                        continue

                    try:
                        for item in os.listdir(higgs_base_dir):
                            item_path = os.path.join(higgs_base_dir, item)
                            if not os.path.isdir(item_path):
                                continue

                            # Check if this subdirectory contains Higgs Audio model files
                            # Must have both generation/ and tokenizer/ subdirs with essential files
                            gen_path = os.path.join(item_path, "generation")
                            tok_path = os.path.join(item_path, "tokenizer")

                            if not (os.path.exists(gen_path) and os.path.exists(tok_path)):
                                continue

                            # Check for essential generation files
                            try:
                                gen_files = os.listdir(gen_path)
                                has_gen_config = "config.json" in gen_files
                                has_gen_model = any(f.endswith(".safetensors") for f in gen_files)
                            except OSError:
                                has_gen_config = False
                                has_gen_model = False

                            # Check for essential tokenizer files
                            try:
                                tok_files = os.listdir(tok_path)
                                has_tok_config = "config.json" in tok_files
                                has_tok_model = any(f.endswith((".pth", ".bin")) for f in tok_files)
                            except OSError:
                                has_tok_config = False
                                has_tok_model = False

                            # Valid if has essential files in both subdirs
                            if has_gen_config and has_gen_model and has_tok_config and has_tok_model:
                                model_name = item
                                local_model_name = f"local:{model_name}"
                                if local_model_name not in found_local_models:
                                    found_local_models.add(local_model_name)

                    except OSError:
                        continue
        except Exception:
            pass

        # Add found local models to the beginning
        for local_model in sorted(found_local_models):
            if local_model not in available:
                available.insert(0, local_model)

        return available if available else ["higgs-audio-v2-3B"]

    @classmethod
    def INPUT_TYPES(cls):
        available_models = cls._get_available_higgs_models()
        
        
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
                "system_prompt": ("STRING", {
                    "default": "Generate audio following instruction.",
                    "multiline": True,
                    "tooltip": "System instruction that guides how Higgs Audio 2 generates speech:\n\n• Default: 'Generate audio following instruction.' - Works for most cases\n• Custom examples:\n  - 'Speak clearly and slowly.' - For clearer pronunciation\n  - 'Generate dramatic, emotional speech.' - For expressive delivery\n  - 'Speak in a calm, professional tone.' - For business/formal content\n\nThis is an advanced parameter - the default usually works best unless you need specific speech characteristics."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "🌡️ Controls speech creativity and randomness:\n\n• 0.0-0.5: Very predictable, robotic speech (not recommended)\n• 0.6-0.8: 🎯 RECOMMENDED - Conservative, natural speech with excellent consistency\n• 1.0: Balanced natural variation but less consistent\n• 1.2-1.5: More expressive, varied pronunciation and pacing\n• 1.8-2.0: Highly creative but potentially unstable\n\n0.8 provides the best balance of natural speech and consistency."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "🎯 Nucleus sampling - controls vocabulary diversity:\n\n• 0.1-0.3: Very limited vocabulary, may sound repetitive\n• 0.5-0.7: 🎯 RECOMMENDED - Focused vocabulary for consistent, clear pronunciation\n• 0.8-0.9: More varied speech patterns but less consistent\n• 0.95-1.0: Maximum vocabulary diversity, may include rare pronunciations\n\n0.6 provides excellent consistency while maintaining natural speech variation."
                }),
                "top_k": ("INT", {
                    "default": 80,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "🔢 Limits vocabulary choices per word:\n\n• -1: Disabled (uses only top_p)\n• 10-30: Very focused, consistent pronunciation\n• 40-60: Balanced consistency and variation\n• 70-90: 🎯 RECOMMENDED - Broader vocabulary pool for natural speech\n• 95-100: Maximum vocabulary freedom, more diverse but potentially inconsistent\n\nWorks with top_p (0.6) to provide good vocabulary range while maintaining consistency."
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "🔤 Maximum token limit - safety cap on generation length:\n\n⚠️ This is a LIMIT, not a target. Model stops when audio is complete OR limit is reached.\n\n• <10 tokens: ⚠️ May cause errors or cut off mid-word\n• 200-500: Safe for short sentences, faster processing\n• 1000-2048: 🎯 RECOMMENDED - Handles most content safely\n• 3000-4096: For very long paragraphs only\n\nFor normal text like 'Hello Bob', 200 vs 2048 makes no difference - same quality and length. Only matters for very short limits (causes truncation) or very long text (needs higher limits)."
                }),
                "force_audio_gen": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "🎵 Force Audio Generation:\n\n• False: 🎯 RECOMMENDED - Model naturally chooses to generate audio tokens\n• True: Force model to generate audio tokens rather than text tokens\n\n⚠️ Only enable if model is generating text instead of audio. Usually not needed as the model should naturally generate audio for TTS requests."
                }),
                "ras_win_len": ("INT", {
                    "default": 7,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "🪟 RAS Window Length - Repetition Avoidance Sampling window size:\n\n• 0: Disable RAS completely (may cause repetitive speech)\n• 3-5: Very strict repetition control (may sound unnatural)\n• 7: 🎯 RECOMMENDED - Good balance of natural speech and repetition control\n• 10-15: Looser repetition control, more natural but may repeat\n• 20: Very loose control, natural speech but potential repetition\n\nRAS prevents the model from repeating the same audio patterns within a sliding window."
                }),
                "ras_max_num_repeat": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "🔄 RAS Max Repetitions - Maximum allowed repetitions within RAS window:\n\n• 1: No repetitions allowed (very strict, may sound choppy)\n• 2: 🎯 RECOMMENDED - Allow minimal repetition for natural speech flow\n• 3: Allow moderate repetition (more natural but some repetition)\n• 4-5: Allow significant repetition (natural speech but potential repetitive patterns)\n\nWorks with RAS Window Length to control speech repetition patterns."
                })
            },
            "optional": {
                "opt_second_narrator": (any_typ, {
                    "tooltip": "Second narrator voice for native multi-speaker modes. Used as SPEAKER1 voice when multi_speaker_mode is set to Native Multi-Speaker. Only needed for native modes, ignored in Custom Character Switching mode. First narrator (from Character Voices or TTS Text) becomes SPEAKER0.\\n\\n💡 TIP: Reference text significantly improves Higgs Audio voice cloning quality - always provide reference text with voice files."
                }),
                "enable_cuda_graphs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "⚡ CUDA Graph Optimization:\n\n• True (High Performance): 55+ tokens/sec generation speed, but memory cannot be unloaded safely (may crash on 'Unload Models'). Use for maximum speed in single-session workflows.\n\n• False (Memory Safe): ~12 tokens/sec generation speed (78% slower), but enables safe memory unloading with 'Unload Models' button. Use for dynamic workflows with multiple models.\n\n⚠️ IMPORTANT: When enabled, DO NOT use 'Unload Models' button - keep model loaded or restart ComfyUI to free memory."
                })
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("tts_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    DESCRIPTION = "Configure Higgs Audio 2 engine for TTS generation with voice cloning. TIP: Reference text significantly improves voice cloning quality."
    
    def create_engine_config(self, model, device, multi_speaker_mode, system_prompt,
                           temperature, top_p, top_k, max_new_tokens, force_audio_gen, 
                           ras_win_len, ras_max_num_repeat, opt_second_narrator=None, 
                           enable_cuda_graphs=True):
        """Create Higgs Audio engine configuration"""
        
        # Validate parameters
        config = {
            "engine_type": "higgs_audio",
            "model": model,
            "device": device,
            "multi_speaker_mode": multi_speaker_mode,
            "system_prompt": system_prompt,
            "temperature": max(0.0, min(2.0, temperature)),
            "top_p": max(0.1, min(1.0, top_p)),
            "top_k": max(-1, min(100, top_k)),
            "max_new_tokens": max(1, min(4096, max_new_tokens)),
            "force_audio_gen": bool(force_audio_gen),
            "ras_win_len": max(0, min(20, ras_win_len)) if ras_win_len > 0 else None,  # None disables RAS
            "ras_max_num_repeat": max(1, min(5, ras_max_num_repeat)),
            "opt_second_narrator": opt_second_narrator,
            "enable_cuda_graphs": bool(enable_cuda_graphs),
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