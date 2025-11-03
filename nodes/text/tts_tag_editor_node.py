"""
🏷️ Multiline TTS Tag Editor

A sophisticated multiline text editor with sidebar controls for TTS-specific tags,
character switching, parameter control, pause insertion, and preset management.

Features:
- Mid-sentence tag insertion ([text [char|seed:2] more text])
- Full undo/redo history with keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z)
- Character and language selection
- Per-segment parameter controls (seed, temperature, cfg, speed, steps, etc.)
- Pause tag insertion with configurable duration
- Preset system for saving/loading common tag combinations
- SRT format support with specialized visualization
- Complete session persistence (text, history, presets, UI state)
"""

import json
from typing import Dict, List, Tuple, Optional, Any


class StringMultilineTagEditor:
    """Multiline string editor with full TTS tag support and state management"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_text"
    CATEGORY = "utils/string"
    OUTPUT_NODE = True

    def __init__(self):
        """Initialize the editor state"""
        self._state = {
            "text": "",
            "history": [],
            "history_index": -1,
            "presets": {},
            "last_character": "",
            "last_language": "",
            "last_seed": 0,
            "last_temperature": 0.7,
            "last_pause_duration": "1s",
            "sidebar_expanded": True,
            "last_cursor_position": 0,
        }
        self._discovered_characters = {}
        self._supported_languages = ["en", "de", "fr", "ja", "es", "it", "pt", "th", "no"]

        # Load persisted state
        self._load_persisted_state()
        # Discover available characters
        self._discover_characters()

    def _load_persisted_state(self):
        """Load saved state from storage (will be implemented in widget)"""
        # This will be called from the frontend to restore widget state
        pass

    def _discover_characters(self):
        """Discover available character voices"""
        try:
            from utils.voice.discovery import VoiceDiscovery
            discovery = VoiceDiscovery()

            # Get all available characters
            characters = discovery.get_available_characters()
            if characters:
                self._discovered_characters = list(characters) if isinstance(characters, set) else characters
                print(f"🎭 TTS Tag Editor: Discovered {len(self._discovered_characters)} character voices")
        except Exception as e:
            print(f"⚠️ TTS Tag Editor: Could not discover characters: {e}")

    def _validate_tag_syntax(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate tag syntax in text"""
        import re

        # Find all tags
        tag_pattern = r'\[([^\]]+)\]'
        tags = re.findall(tag_pattern, text)

        for tag_content in tags:
            # Check for mismatched brackets
            if tag_content.count('[') != tag_content.count(']'):
                return False, f"Mismatched brackets in tag: [{tag_content}]"

            # Validate parameter syntax if present
            if '|' in tag_content:
                parts = tag_content.split('|')
                for i, part in enumerate(parts):
                    if i == 0:
                        # First part is character (or language:character)
                        continue
                    # Validate parameter format
                    if ':' not in part:
                        return False, f"Invalid parameter syntax: {part} (expected format: param:value)"

                    param_name, param_value = part.split(':', 1)
                    # Validate parameter name (should not be empty)
                    if not param_name.strip():
                        return False, f"Empty parameter name in {part}"

        return True, None

    def _parse_existing_tags(self, text: str) -> List[Dict[str, Any]]:
        """Parse existing tags from text"""
        import re

        tags = []
        tag_pattern = r'\[([^\]]+)\]'

        for match in re.finditer(tag_pattern, text):
            tag_content = match.group(1)
            tag_dict = {
                "full": f"[{tag_content}]",
                "position": match.start(),
                "character": "",
                "language": "",
                "parameters": {}
            }

            # Parse character and parameters
            parts = tag_content.split('|')

            # First part: character or language:character
            first_part = parts[0].strip()
            if ':' in first_part and not '.' in first_part:  # language:character (not decimal numbers)
                lang, char = first_part.split(':', 1)
                tag_dict["language"] = lang.strip()
                tag_dict["character"] = char.strip()
            else:
                tag_dict["character"] = first_part

            # Remaining parts: parameters
            for part in parts[1:]:
                if ':' in part:
                    param_name, param_value = part.split(':', 1)
                    tag_dict["parameters"][param_name.strip().lower()] = param_value.strip()

            tags.append(tag_dict)

        return tags

    def _add_to_history(self, text: str):
        """Add text state to undo/redo history"""
        # Remove any future history if we've undone something
        if self._state["history_index"] < len(self._state["history"]) - 1:
            self._state["history"] = self._state["history"][:self._state["history_index"] + 1]

        # Add new state
        self._state["history"].append(text)
        self._state["history_index"] = len(self._state["history"]) - 1

        # Keep history size reasonable (limit to 100 states)
        if len(self._state["history"]) > 100:
            self._state["history"] = self._state["history"][-100:]
            self._state["history_index"] = len(self._state["history"]) - 1

    def _undo(self) -> str:
        """Undo to previous state"""
        if self._state["history_index"] > 0:
            self._state["history_index"] -= 1
            return self._state["history"][self._state["history_index"]]
        return self._state["text"]

    def _redo(self) -> str:
        """Redo to next state"""
        if self._state["history_index"] < len(self._state["history"]) - 1:
            self._state["history_index"] += 1
            return self._state["history"][self._state["history_index"]]
        return self._state["text"]

    def _insert_tag_at_position(self, text: str, tag: str, position: int, wrap_selection: bool = False,
                               selection_start: int = -1, selection_end: int = -1) -> str:
        """Insert tag at specified position, optionally wrapping selection"""
        if wrap_selection and selection_start >= 0 and selection_end >= 0:
            # Wrap selected text with tag
            before = text[:selection_start]
            selected = text[selection_start:selection_end]
            after = text[selection_end:]

            # Insert tag before selected text
            return f"{before}{tag} {selected}{after}"
        else:
            # Insert tag at caret position
            return f"{text[:position]}{tag} {text[position:]}"

    def _save_preset(self, preset_name: str, tag_content: str) -> bool:
        """Save a preset"""
        if not preset_name.strip():
            return False

        self._state["presets"][preset_name] = {
            "tag": tag_content,
            "created": str(self._get_timestamp())
        }
        return True

    def _load_preset(self, preset_name: str) -> Optional[str]:
        """Load a preset"""
        if preset_name in self._state["presets"]:
            return self._state["presets"][preset_name]["tag"]
        return None

    def _delete_preset(self, preset_name: str) -> bool:
        """Delete a preset"""
        if preset_name in self._state["presets"]:
            del self._state["presets"][preset_name]
            return True
        return False

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _validate_srt_format(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate if text is in SRT format"""
        import re

        # SRT pattern: index\ntimestamp\ntext
        srt_pattern = r'^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s*\n'

        lines = text.strip().split('\n')
        if len(lines) < 3:
            return False, None

        # Check if it looks like SRT
        is_srt = bool(re.match(srt_pattern, text))
        return is_srt, None

    def _serialize_state(self) -> str:
        """Serialize complete state for persistence"""
        return json.dumps(self._state, indent=2)

    def _deserialize_state(self, state_json: str):
        """Deserialize state from persistence"""
        try:
            loaded_state = json.loads(state_json)
            self._state.update(loaded_state)
        except Exception as e:
            print(f"⚠️ Failed to deserialize state: {e}")

    def get_available_characters(self) -> List[str]:
        """Return list of available character voices for the widget"""
        return self._discovered_characters if isinstance(self._discovered_characters, list) else list(self._discovered_characters)

    def process_text(self, text: str) -> Tuple[str]:
        """Main processing function - returns text from widget"""
        # Receives the text value from the widget through ComfyUI's input system
        # The widget's getValue() provides the current textarea content
        return (text,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "StringMultilineTagEditor": StringMultilineTagEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringMultilineTagEditor": "🏷️ Multiline TTS Tag Editor"
}
