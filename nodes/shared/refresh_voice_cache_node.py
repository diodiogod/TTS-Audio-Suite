"""
Refresh Voice Cache Node

For workflows that create voice reference files during execution.
This node forces a rescan of voice folders (voices + character cache + alias map)
so downstream TTS nodes can resolve newly-created [Character] tags without UI refresh.
"""

import os
import sys
import importlib.util
from typing import Any, Tuple


# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Load base_node module directly (matches repo pattern when loaded via importlib)
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

BaseTTSNode = base_module.BaseTTSNode


from utils.voice.discovery import get_available_voices, get_available_characters


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class RefreshVoiceCacheNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "♻️ Refresh Voice Cache"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "signal": (
                    any_typ,
                    {
                        "tooltip": "Primary signal input. Connect this to the output of a node that creates new voice files (e.g., Voice Designer) to ensure voices are refreshed after file creation."  # noqa: E501
                    },
                ),
            },
            "optional": {
                "signal2": (
                    any_typ,
                    {
                        "default": None,
                        "tooltip": "Secondary passthrough signal. Use this to chain execution order: connect your ENGINE/config here, and connect signal2_out to downstream TTS nodes. This ensures TTS nodes run after voice cache refresh."  # noqa: E501
                    },
                ),
                "force_refresh": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Force a full rescan of voice folders and alias map.",  # noqa: E501
                    },
                ),
            },
        }

    RETURN_TYPES = (any_typ, any_typ, "STRING")
    RETURN_NAMES = ("signal", "signal2", "info")
    FUNCTION = "refresh"
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"

    def refresh(self, signal: Any, signal2: Any = None, force_refresh: bool = True) -> Tuple[Any, Any, str]:
        try:
            if force_refresh:
                voices = get_available_voices(force_refresh=True)
                chars = get_available_characters(force_refresh=True)
            else:
                voices = get_available_voices(force_refresh=False)
                chars = get_available_characters(force_refresh=False)

            voice_count = max(0, len(voices) - 1)  # exclude "none"
            info = f"✅ Voice cache refreshed: {voice_count} voices, {len(chars)} characters"
            print(f"♻️ Refresh Voice Cache: {info}")
            return signal, signal2, info
        except Exception as e:
            msg = f"❌ Voice cache refresh failed: {e}"
            print(f"♻️ Refresh Voice Cache: {msg}")
            return signal, signal2, msg


NODE_CLASS_MAPPINGS = {
    "RefreshVoiceCacheNode": RefreshVoiceCacheNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RefreshVoiceCacheNode": "♻️ Refresh Voice Cache",
}
