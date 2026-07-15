"""Explicit output node for persisting an opt_narrator as a character voice."""

import os
import sys

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.voice.character_saver import save_character_voice


class SaveCharacterVoiceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "opt_narrator": (
                    "NARRATOR_VOICE",
                    {
                        "tooltip": (
                            "Character voice to save. Connect opt_narrator from Character Voices "
                            "or a Voice Designer. Audio and the exact transcription are reused automatically."
                        )
                    },
                ),
                "character_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Filename and [Character] tag name for the saved voice.",
                    },
                ),
                "overwrite_character": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Overwrite existing character files. When disabled, an existing name is saved "
                            "as name_1, name_2, etc."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("NARRATOR_VOICE", "STRING", "STRING")
    RETURN_NAMES = ("opt_narrator", "character_name", "save_info")
    FUNCTION = "save_character"
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"
    OUTPUT_NODE = True

    def save_character(self, opt_narrator, character_name: str, overwrite_character: bool):
        result = save_character_voice(
            opt_narrator=opt_narrator,
            character_name=character_name,
            overwrite_character=overwrite_character,
        )
        print(f"💾 Save Character Voice: {result.info}")
        return result.opt_narrator, result.character_name, result.info


NODE_CLASS_MAPPINGS = {"SaveCharacterVoiceNode": SaveCharacterVoiceNode}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveCharacterVoiceNode": "💾 Save Character Voice"}
