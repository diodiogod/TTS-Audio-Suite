"""
MiniMax Cloud TTS Engine Configuration Node

Provides MiniMax Cloud TTS configuration for TTS Audio Suite.
Requires a MINIMAX_API_KEY environment variable.

MiniMax T2A v2 API: https://platform.minimaxi.com/document/T2A%20V2
Models: speech-2.8-hd (high quality), speech-2.8-turbo (low latency)
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


class MiniMaxTTSEngineNode(BaseTTSNode):
    """
    MiniMax Cloud TTS engine configuration node.

    Uses the MiniMax T2A v2 API for cloud-based text-to-speech.
    Requires MINIMAX_API_KEY environment variable.
    """

    @classmethod
    def NAME(cls):
        return "MiniMax Cloud TTS Engine"

    @classmethod
    def INPUT_TYPES(cls):
        from engines.adapters.minimax_tts_adapter import MINIMAX_VOICE_IDS, MINIMAX_MODELS

        return {
            "required": {
                "model": (MINIMAX_MODELS, {
                    "default": "speech-2.8-hd",
                    "tooltip": (
                        "MiniMax TTS model:\n"
                        "- speech-2.8-hd: High-definition quality, best for final output\n"
                        "- speech-2.8-turbo: Lower latency, good for previews"
                    ),
                }),
                "voice_id": (MINIMAX_VOICE_IDS, {
                    "default": "English_Graceful_Lady",
                    "tooltip": (
                        "Built-in voice to use for speech synthesis.\n"
                        "12 voices available across English and multilingual styles."
                    ),
                }),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Speech speed multiplier. 1.0 = normal, <1.0 = slower, >1.0 = faster.",
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "MiniMax API key. If empty, reads from MINIMAX_API_KEY environment variable.\n"
                        "Get your API key at https://platform.minimaxi.com/"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/Engines"

    def create_engine_adapter(self, model: str, voice_id: str, speed: float,
                              api_key: str = ""):
        """Create MiniMax Cloud TTS engine configuration."""
        try:
            # Verify adapter is importable
            from engines.adapters.minimax_tts_adapter import MiniMaxTTSAdapter
            _ = MiniMaxTTSAdapter  # silence unused

            config = {
                "engine_type": "minimax_tts",
                "model": model,
                "voice_id": voice_id,
                "speed": float(speed),
            }

            # Only store API key in config if explicitly provided
            if api_key:
                config["api_key"] = api_key

            # Check API key availability
            resolved_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
            key_source = "config" if api_key else ("env" if resolved_key else "missing")

            print(f"MiniMax Cloud TTS Engine: model={model}, voice={voice_id}, speed={speed:.2f}")
            if key_source == "missing":
                print("   \u26a0\ufe0f WARNING: No API key found. Set MINIMAX_API_KEY environment variable.")
            else:
                print(f"   API key source: {key_source}")

            engine_data = {
                "engine_type": "minimax_tts",
                "config": config,
                "adapter_class": "MiniMaxTTSAdapter",
            }

            return (engine_data,)

        except Exception as e:
            print(f"ERROR: MiniMax Cloud TTS Engine error: {e}")
            import traceback
            traceback.print_exc()

            error_config = {
                "engine_type": "minimax_tts",
                "config": {
                    "model": model,
                    "voice_id": voice_id,
                    "error": str(e),
                },
                "adapter_class": "MiniMaxTTSAdapter",
            }
            return (error_config,)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "MiniMaxTTSEngineNode": MiniMaxTTSEngineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxTTSEngineNode": "MiniMax Cloud TTS Engine",
}
