"""
Voice Discovery Utility for F5-TTS and ChatterBox Integration
Enhanced voice file discovery with dual folder support, smart text file priority, and character mapping
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
try:
    import folder_paths
except ImportError:
    folder_paths = None


class VoiceDiscovery:
    """
    Enhanced voice discovery system for F5-TTS and ChatterBox nodes.
    
    Features:
    - Dual folder support: models/voices/ and voices_examples/
    - Recursive folder scanning with nested directory support
    - Smart text file priority: .reference.txt > .txt
    - Character mapping for multi-character TTS
    - Support for both F5TTS (with text) and ChatterBox (audio only)
    - Performance caching to avoid repeated filesystem scans
    - Backward compatibility with existing flat structure
    """
    
    def __init__(self):
        self._cache = {}
        self._cache_valid = False
        self._character_cache = {}
        self._character_cache_valid = False
        self._character_aliases = {}
        self._aliases_valid = False
        
        # Initialize character discovery on first import
        self._initialize_character_discovery()
    
    def _initialize_character_discovery(self):
        """Initialize character discovery system on startup."""
        try:
            # Force character cache refresh to discover characters immediately
            self._refresh_character_cache()
            # Load character aliases after characters are discovered
            self._refresh_character_aliases()
            if len(self._character_cache) > 0:
                alias_count = len(self._character_aliases)
                if alias_count > 0:
                    print(f"🎭 Character voices: Found {len(self._character_cache)} characters, {alias_count} aliases")
                else:
                    print(f"🎭 Character voices: Found {len(self._character_cache)} characters")
        except Exception as e:
            print(f"⚠️ Character discovery failed: {e}")
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voice files with companion text files.
        
        Returns:
            List of voice file paths relative to their base directories.
            Format: ["voice.wav", "folder/voice.wav", "voices_examples/speaker.wav"]
        """
        if not self._cache_valid:
            self._refresh_cache()
        
        return ["none"] + sorted(self._cache.keys())
    
    def get_voice_info(self, voice_key: str) -> Optional[Dict[str, str]]:
        """
        Get detailed information about a voice file.
        
        Args:
            voice_key: Voice key from get_available_voices()
            
        Returns:
            Dict with 'audio_path', 'text_path', 'text_content', 'source_folder'
        """
        if voice_key == "none" or not self._cache_valid:
            return None
            
        return self._cache.get(voice_key)
    
    def load_voice_reference(self, voice_key: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Load voice audio path and reference text content.
        
        Args:
            voice_key: Voice key from get_available_voices()
            
        Returns:
            Tuple of (audio_path, reference_text) or (None, None) if not found
        """
        if voice_key == "none":
            return None, None
            
        voice_info = self.get_voice_info(voice_key)
        if not voice_info:
            return None, None
            
        return voice_info['audio_path'], voice_info['text_content']
    
    def _refresh_cache(self):
        """Refresh the voice cache by scanning all supported directories."""
        self._cache.clear()
        
        # Scan models/voices/ directory (existing location)
        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir and os.path.exists(models_voices_dir):
            self._scan_directory(models_voices_dir, "models/voices", "")
        
        # Scan voices_examples/ directory (new location in custom_node)
        voices_examples_dir = self._get_voices_examples_dir()
        if voices_examples_dir and os.path.exists(voices_examples_dir):
            self._scan_directory(voices_examples_dir, "voices_examples", "voices_examples")
        
        self._cache_valid = True
        # print(f"🎤 Voice Discovery: Found {len(self._cache)} voices across all locations")
    
    def _get_models_voices_dir(self) -> Optional[str]:
        """Get the ComfyUI models/voices directory path."""
        try:
            if folder_paths is None:
                return None
            models_dir = folder_paths.models_dir
            return os.path.join(models_dir, "voices")
        except:
            return None
    
    def _get_voices_examples_dir(self) -> Optional[str]:
        """Get the custom_node voices_examples directory path."""
        try:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.dirname(__file__))
            return os.path.join(current_dir, "voices_examples")
        except:
            return None
    
    def _scan_directory(self, base_dir: str, source_name: str, prefix: str):
        """
        Recursively scan a directory for voice files with text companions.
        
        Args:
            base_dir: Base directory to scan
            source_name: Name for logging (e.g., "models/voices")
            prefix: Prefix for voice keys (e.g., "voices_examples")
        """
        try:
            for root, dirs, files in os.walk(base_dir):
                # Get relative path from base directory
                rel_path = os.path.relpath(root, base_dir)
                if rel_path == ".":
                    rel_path = ""
                
                # Filter audio files
                audio_files = self._filter_audio_files(files)
                
                for audio_file in audio_files:
                    audio_path = os.path.join(root, audio_file)
                    
                    # Find companion text file with priority
                    text_path, text_content = self._find_companion_text(audio_path)
                    
                    if text_path and text_content:
                        # Create voice key for dropdown
                        if rel_path:
                            voice_key = f"{prefix}/{rel_path}/{audio_file}" if prefix else f"{rel_path}/{audio_file}"
                        else:
                            voice_key = f"{prefix}/{audio_file}" if prefix else audio_file
                        
                        # Clean up voice key (remove double slashes, etc.)
                        voice_key = voice_key.replace("//", "/").strip("/")
                        
                        self._cache[voice_key] = {
                            'audio_path': audio_path,
                            'text_path': text_path,
                            'text_content': text_content,
                            'source_folder': source_name,
                            'relative_path': rel_path
                        }
                        
        except Exception as e:
            print(f"⚠️ Voice Discovery: Error scanning {source_name}: {e}")
    
    def _filter_audio_files(self, files: List[str]) -> List[str]:
        """Filter files to only include audio files."""
        try:
            # Use ComfyUI's built-in audio filtering if available
            if folder_paths is not None:
                return folder_paths.filter_files_content_types(files, ["audio", "video"])
        except:
            pass
        
        # Fallback to manual filtering
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        return [f for f in files if Path(f).suffix.lower() in audio_extensions]
    
    def _find_companion_text(self, audio_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find companion text file with smart priority system.
        
        Priority:
        1. audio_name.reference.txt (F5-TTS reference text)
        2. audio_name.txt (fallback if no .reference.txt)
        
        Args:
            audio_path: Full path to audio file
            
        Returns:
            Tuple of (text_file_path, text_content) or (None, None)
        """
        audio_stem = Path(audio_path).stem
        audio_dir = os.path.dirname(audio_path)
        
        # Priority 1: Look for .reference.txt file
        reference_txt = os.path.join(audio_dir, f"{audio_stem}.reference.txt")
        if os.path.isfile(reference_txt):
            try:
                with open(reference_txt, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only use if not empty
                        return reference_txt, content
            except Exception as e:
                print(f"⚠️ Voice Discovery: Error reading {reference_txt}: {e}")
        
        # Priority 2: Look for regular .txt file
        regular_txt = os.path.join(audio_dir, f"{audio_stem}.txt")
        if os.path.isfile(regular_txt):
            try:
                with open(regular_txt, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Only use if not empty
                        return regular_txt, content
            except Exception as e:
                print(f"⚠️ Voice Discovery: Error reading {regular_txt}: {e}")
        
        return None, None
    
    def invalidate_cache(self):
        """Invalidate the cache to force refresh on next access."""
        self._cache_valid = False
        self._cache.clear()
        self._character_cache_valid = False
        self._character_cache.clear()
        self._aliases_valid = False
        self._character_aliases.clear()
    
    def get_available_characters(self) -> Set[str]:
        """
        Get set of available character names from voice folders.
        
        Returns:
            Set of character names found in voice directories
        """
        if not self._character_cache_valid:
            self._refresh_character_cache()
        
        return set(self._character_cache.keys())
    
    def get_character_voice_info(self, character_name: str, engine_type: str = "f5tts") -> Optional[Dict[str, str]]:
        """
        Get voice information for a specific character.
        
        Args:
            character_name: Name of the character
            engine_type: "f5tts" or "chatterbox" - determines if text is required
            
        Returns:
            Dict with voice info or None if not found
        """
        if not self._character_cache_valid:
            self._refresh_character_cache()
        
        character_info = self._character_cache.get(character_name.lower())
        if not character_info:
            return None
        
        # For ChatterBox, we don't require text files
        if engine_type == "chatterbox":
            return {
                'audio_path': character_info['audio_path'],
                'character_name': character_name,
                'source_folder': character_info['source_folder']
            }
        
        # For F5TTS, we need both audio and text
        if character_info.get('text_content'):
            return character_info
        
        return None
    
    def load_character_voice(self, character_name: str, engine_type: str = "f5tts") -> Tuple[Optional[str], Optional[str]]:
        """
        Load character voice for TTS generation.
        
        Args:
            character_name: Name of the character
            engine_type: "f5tts" (needs text) or "chatterbox" (audio only)
            
        Returns:
            Tuple of (audio_path, reference_text) or (None, None) if not found
        """
        voice_info = self.get_character_voice_info(character_name, engine_type)
        if not voice_info:
            return None, None
        
        audio_path = voice_info['audio_path']
        text_content = voice_info.get('text_content', "")
        
        return audio_path, text_content
    
    def get_character_mapping(self, characters: List[str], engine_type: str = "f5tts") -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """
        Get voice mapping for multiple characters.
        
        Args:
            characters: List of character names to look up
            engine_type: "f5tts" or "chatterbox"
            
        Returns:
            Dict mapping character names to (audio_path, reference_text) tuples
        """
        mapping = {}
        
        for character in characters:
            audio_path, text_content = self.load_character_voice(character, engine_type)
            mapping[character] = (audio_path, text_content)
        
        return mapping
    
    def _refresh_character_cache(self):
        """Refresh the character cache by scanning for character-organized voices."""
        self._character_cache.clear()
        
        # Ensure main voice cache is valid
        if not self._cache_valid:
            self._refresh_cache()
        
        # Scan voices_examples/ directory for character folders
        voices_examples_dir = self._get_voices_examples_dir()
        if voices_examples_dir and os.path.exists(voices_examples_dir):
            self._scan_character_directories(voices_examples_dir, "voices_examples")
        
        # Scan models/voices/ directory for character folders
        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir and os.path.exists(models_voices_dir):
            self._scan_character_directories(models_voices_dir, "models/voices")
        
        self._character_cache_valid = True
    
    def _refresh_character_aliases(self):
        """Refresh character aliases from alias map files."""
        self._character_aliases.clear()
        
        # Load aliases from voices_examples folder first (lower priority)
        voices_examples_dir = self._get_voices_examples_dir()
        if voices_examples_dir:
            self._load_alias_file(voices_examples_dir, "voices_examples")
        
        # Load aliases from models/voices folder (higher priority - overrides examples)
        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir:
            self._load_alias_file(models_voices_dir, "models/voices")
        
        self._aliases_valid = True
    
    def _load_alias_file(self, base_dir: str, source_name: str):
        """Load character aliases from a character_alias_map.json file."""
        alias_file = os.path.join(base_dir, "character_alias_map.json")
        
        if not os.path.exists(alias_file):
            return
        
        try:
            with open(alias_file, 'r', encoding='utf-8') as f:
                aliases = json.load(f)
            
            if not isinstance(aliases, dict):
                print(f"⚠️ Character Aliases: Invalid format in {alias_file} - must be a JSON object")
                return
            
            loaded_count = 0
            for alias, target in aliases.items():
                if isinstance(alias, str) and isinstance(target, str):
                    # Convert to lowercase for consistent matching
                    self._character_aliases[alias.lower()] = target.lower()
                    loaded_count += 1
                else:
                    print(f"⚠️ Character Aliases: Invalid alias entry '{alias}' -> '{target}' in {alias_file}")
            
            if loaded_count > 0:
                pass  # print(f"🗂️ Character Aliases: Loaded {loaded_count} aliases from {source_name}")
        except json.JSONDecodeError as e:
            print(f"⚠️ Character Aliases: Invalid JSON in {alias_file}: {e}")
        except Exception as e:
            print(f"⚠️ Character Aliases: Error reading {alias_file}: {e}")
    
    def resolve_character_alias(self, character_name: str) -> str:
        """
        Resolve a character name through the alias system.
        
        Args:
            character_name: Character name or alias to resolve
            
        Returns:
            Resolved character name, or original if no alias found
        """
        if not self._aliases_valid:
            self._refresh_character_aliases()
        
        normalized_name = character_name.lower()
        return self._character_aliases.get(normalized_name, normalized_name)
    
    def get_character_aliases(self) -> Dict[str, str]:
        """
        Get all character aliases.
        
        Returns:
            Dictionary of alias -> target character mappings
        """
        if not self._aliases_valid:
            self._refresh_character_aliases()
        
        return self._character_aliases.copy()
    
    def _scan_character_directories(self, base_dir: str, source_name: str):
        """
        Scan for character voices in both directory structure and flat files.
        
        Supported structures:
        1. Directory structure:
           base_dir/
           ├── narrator/
           │   ├── main.wav
           │   └── main.reference.txt (for F5TTS)
           └── alice/
               ├── alice.wav
               └── alice.reference.txt (for F5TTS)
        
        2. Flat file structure:
           base_dir/
           ├── female_01.wav
           ├── female_01.reference.txt (for F5TTS)
           ├── male_01.wav
           └── male_01.reference.txt (for F5TTS)
        """
        try:
            # First, scan for character directories
            self._scan_character_folders(base_dir, source_name)
            
            # Then, scan for flat character files
            self._scan_flat_character_files(base_dir, source_name)
                    
        except Exception as e:
            print(f"⚠️ Character Discovery: Error scanning {source_name}: {e}")
    
    def _scan_character_folders(self, base_dir: str, source_name: str):
        """Scan for character-organized voice directories."""
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                
                # Skip files, only process directories
                if not os.path.isdir(item_path):
                    continue
                
                # Character name is the directory name
                character_name = item.lower()
                
                # Find audio files in this character directory
                try:
                    files = os.listdir(item_path)
                    audio_files = self._filter_audio_files(files)
                    
                    if not audio_files:
                        continue
                    
                    # Use the first audio file found
                    audio_file = audio_files[0]
                    audio_path = os.path.join(item_path, audio_file)
                    
                    # Look for companion text file
                    text_path, text_content = self._find_companion_text(audio_path)
                    
                    # Store character info
                    self._character_cache[character_name] = {
                        'audio_path': audio_path,
                        'text_path': text_path,
                        'text_content': text_content or "",
                        'source_folder': source_name,
                        'character_directory': item_path
                    }
                    
                        
                except Exception as e:
                    print(f"⚠️ Error scanning character directory {item_path}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Character Discovery: Error scanning character folders in {source_name}: {e}")
    
    def _scan_flat_character_files(self, base_dir: str, source_name: str):
        """Scan for character voice files in flat structure (e.g., female_01.wav, male_01.wav)."""
        try:
            files = os.listdir(base_dir)
            audio_files = self._filter_audio_files(files)
            
            for audio_file in audio_files:
                audio_path = os.path.join(base_dir, audio_file)
                
                # Skip if this is a file inside a character directory (already processed)
                if os.path.dirname(audio_path) != base_dir:
                    continue
                
                # Extract character name from filename (remove extension)
                character_name = Path(audio_file).stem.lower()
                
                # Skip if this character was already found in a directory
                if character_name in self._character_cache:
                    continue
                
                # Look for companion text file
                text_path, text_content = self._find_companion_text(audio_path)
                
                # Store character info
                self._character_cache[character_name] = {
                    'audio_path': audio_path,
                    'text_path': text_path,
                    'text_content': text_content or "",
                    'source_folder': source_name,
                    'character_directory': None  # Flat file, no directory
                }
                    
        except Exception as e:
            print(f"⚠️ Character Discovery: Error scanning flat files in {source_name}: {e}")
    
    def has_character_support(self) -> bool:
        """
        Check if any character voices are available.
        
        Returns:
            True if character voices are found
        """
        return len(self.get_available_characters()) > 0
    
    def get_character_statistics(self) -> Dict[str, any]:
        """
        Get statistics about available characters.
        
        Returns:
            Dict with character statistics
        """
        if not self._character_cache_valid:
            self._refresh_character_cache()
        
        total_characters = len(self._character_cache)
        f5tts_ready = sum(1 for info in self._character_cache.values() if info.get('text_content'))
        chatterbox_ready = sum(1 for info in self._character_cache.values() if info.get('audio_path'))
        
        return {
            'total_characters': total_characters,
            'f5tts_ready': f5tts_ready,
            'chatterbox_ready': chatterbox_ready,
            'characters': list(self._character_cache.keys())
        }


# Global instance for use across F5-TTS nodes
voice_discovery = VoiceDiscovery()


def get_available_voices() -> List[str]:
    """Convenience function to get available voices."""
    return voice_discovery.get_available_voices()


def load_voice_reference(voice_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Convenience function to load voice reference."""
    return voice_discovery.load_voice_reference(voice_key)


def invalidate_voice_cache():
    """Convenience function to invalidate voice cache."""
    voice_discovery.invalidate_cache()


# Character voice convenience functions

def get_available_characters() -> Set[str]:
    """Convenience function to get available characters."""
    return voice_discovery.get_available_characters()


def load_character_voice(character_name: str, engine_type: str = "f5tts") -> Tuple[Optional[str], Optional[str]]:
    """Convenience function to load character voice."""
    return voice_discovery.load_character_voice(character_name, engine_type)


def get_character_mapping(characters: List[str], engine_type: str = "f5tts") -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Convenience function to get character voice mapping."""
    return voice_discovery.get_character_mapping(characters, engine_type)


def has_character_support() -> bool:
    """Convenience function to check if character voices are available."""
    return voice_discovery.has_character_support()