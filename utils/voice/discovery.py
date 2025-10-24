"""
Voice Discovery Utility for F5-TTS and ChatterBox Integration
Enhanced voice file discovery with dual folder support, smart text file priority, and character mapping
Includes persistent caching for faster startup times.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union
try:
    import folder_paths
    from utils.models.extra_paths import get_all_voices_paths
except ImportError:
    folder_paths = None
    get_all_voices_paths = None

from utils.voice.cache_manager import VoiceDiscoveryCacheManager


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
        self._character_language_defaults = {}
        self._aliases_valid = False

        # Initialize cache manager for persistent caching
        self._cache_manager = VoiceDiscoveryCacheManager()
        self._voice_dirs_tracked = self._get_all_voice_dirs()

        # Defer character discovery to first access (non-blocking startup)
        self._initialized = False
    
    def _ensure_initialized(self):
        """
        Lazy initialization on first access.
        - Loads from cache immediately (fast)
        - Starts background refresh thread to update cache after ComfyUI loads
        """
        if self._initialized:
            return

        self._initialized = True

        # Try loading from cache first (instant - no filesystem scan)
        if self._cache_manager.is_cache_valid():
            try:
                cached_data = self._cache_manager.load_cache()
                if cached_data:
                    self._character_cache = cached_data.get('character_cache', {})
                    self._character_aliases = cached_data.get('character_aliases', {})
                    self._character_language_defaults = cached_data.get('character_language_defaults', {})
                    self._character_cache_valid = True
                    self._aliases_valid = True

                    if len(self._character_cache) > 0:
                        alias_count = len(self._character_aliases)
                        if alias_count > 0:
                            print(f"[TTS Audio Suite] 🎭 Character voices: Found {len(self._character_cache)} characters, {alias_count} aliases (cached)")
                        else:
                            print(f"[TTS Audio Suite] 🎭 Character voices: Found {len(self._character_cache)} characters (cached)")
                        print(f"[TTS Audio Suite] 🔄 Updating character discovery in background...")

                    # Start background refresh after ComfyUI loads
                    self._cache_manager.start_background_refresh(self._get_fresh_cache_data)
                    return
            except Exception:
                pass  # Fallback to full scan

        # No cache - do full discovery now
        try:
            self._refresh_character_cache()
            self._refresh_character_aliases()

            # Save to persistent cache for next startup
            cache_data = self._get_fresh_cache_data()
            self._cache_manager.save_cache(cache_data)

            if len(self._character_cache) > 0:
                alias_count = len(self._character_aliases)
                if alias_count > 0:
                    print(f"[TTS Audio Suite] 🎭 Character voices: Found {len(self._character_cache)} characters, {alias_count} aliases")
                else:
                    print(f"[TTS Audio Suite] 🎭 Character voices: Found {len(self._character_cache)} characters")

            # Start background refresh for updates
            self._cache_manager.start_background_refresh(self._get_fresh_cache_data)

        except Exception as e:
            print(f"⚠️ Character discovery failed: {e}")

    def _get_fresh_cache_data(self) -> Tuple[Dict, bool]:
        """
        Get fresh cache data by rescanning directories.

        Returns:
            Tuple of (cache_data, was_updated) where was_updated indicates if content changed
        """
        try:
            # Store old data to compare
            old_chars = set(self._character_cache.keys())
            old_aliases = set(self._character_aliases.keys())

            # Rescan
            self._refresh_character_cache()
            self._refresh_character_aliases()

            # Check if anything changed
            new_chars = set(self._character_cache.keys())
            new_aliases = set(self._character_aliases.keys())
            was_updated = (old_chars != new_chars) or (old_aliases != new_aliases)

            cache_data = {
                'character_cache': self._character_cache,
                'character_aliases': self._character_aliases,
                'character_language_defaults': self._character_language_defaults
            }

            # Print update message if cache changed
            if was_updated:
                char_diff = len(new_chars) - len(old_chars)
                alias_diff = len(new_aliases) - len(old_aliases)
                msg_parts = []
                if char_diff != 0:
                    msg_parts.append(f"{char_diff:+d} characters" if char_diff > 0 else f"{char_diff} characters")
                if alias_diff != 0:
                    msg_parts.append(f"{alias_diff:+d} aliases" if alias_diff > 0 else f"{alias_diff} aliases")
                if msg_parts:
                    print(f"🎭 Character voices updated: {', '.join(msg_parts)}")

            return cache_data, was_updated
        except Exception as e:
            return {}, False

    def _get_all_voice_dirs(self) -> List[str]:
        """Get list of all voice directories to track for caching."""
        dirs = []

        # Add standard directories
        voices_examples_dir = self._get_voices_examples_dir()
        if voices_examples_dir:
            dirs.append(voices_examples_dir)

        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir:
            dirs.append(models_voices_dir)

        models_tts_voices_dir = self._get_models_tts_voices_dir()
        if models_tts_voices_dir:
            dirs.append(models_tts_voices_dir)

        # Add extra paths if available
        if get_all_voices_paths:
            try:
                extra_paths = get_all_voices_paths()
                dirs.extend(extra_paths)
            except Exception:
                pass

        return dirs
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voice files with companion text files.

        Returns:
            List of voice file paths relative to their base directories.
            Format: ["voice.wav", "folder/voice.wav", "voices_examples/speaker.wav"]
        """
        # Lazy initialization on first access
        self._ensure_initialized()

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
        
        # Scan models/voices/ directory (primary location)
        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir and os.path.exists(models_voices_dir):
            self._scan_directory(models_voices_dir, "models/voices", "")
        
        # Scan models/TTS/voices/ directory (fallback location)
        models_tts_voices_dir = self._get_models_tts_voices_dir()
        if models_tts_voices_dir and os.path.exists(models_tts_voices_dir):
            self._scan_directory(models_tts_voices_dir, "models/TTS/voices", "TTS/voices")
        
        # Scan voices_examples/ directory (bundled examples)
        voices_examples_dir = self._get_voices_examples_dir()
        if voices_examples_dir and os.path.exists(voices_examples_dir):
            self._scan_directory(voices_examples_dir, "voices_examples", "voices_examples")
        
        # Scan any additional extra_model_paths.yaml configured voices directories
        if get_all_voices_paths:
            try:
                extra_voices_paths = get_all_voices_paths()
                # Keep track of already scanned paths to avoid duplicates
                scanned_paths = set()
                if models_voices_dir:
                    scanned_paths.add(os.path.normpath(models_voices_dir))
                if models_tts_voices_dir:
                    scanned_paths.add(os.path.normpath(models_tts_voices_dir))

                for extra_path in extra_voices_paths:
                    # Skip paths we already scanned above (normalize paths for comparison)
                    normalized_extra = os.path.normpath(extra_path)
                    if normalized_extra in scanned_paths:
                        continue
                    if os.path.exists(extra_path):
                        # Create a clean prefix from the path for organization
                        path_name = os.path.basename(os.path.normpath(extra_path))
                        parent_name = os.path.basename(os.path.dirname(extra_path))
                        if parent_name and parent_name != "models":
                            prefix = f"{parent_name}/{path_name}"
                        else:
                            prefix = path_name
                        self._scan_directory(extra_path, f"extra:{extra_path}", prefix)
                        scanned_paths.add(normalized_extra)
            except Exception:
                # Silently handle any extra paths errors to avoid breaking main functionality
                pass
        
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
            # Get the project root directory (go up from utils/voice/ to root)
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            return os.path.join(current_dir, "voices_examples")
        except:
            return None
    
    def _get_models_tts_voices_dir(self) -> Optional[str]:
        """Get the ComfyUI models/TTS/voices directory path."""
        try:
            if folder_paths is None:
                return None
            models_dir = folder_paths.models_dir
            return os.path.join(models_dir, "TTS", "voices")
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
        self._character_language_defaults.clear()
    
    def get_available_characters(self) -> Set[str]:
        """
        Get set of available character names from voice folders.

        Returns:
            Set of character names found in voice directories
        """
        # Lazy initialization on first access
        self._ensure_initialized()

        if not self._character_cache_valid:
            self._refresh_character_cache()

        return set(self._character_cache.keys())
    
    def get_character_voice_info(self, character_name: str, engine_type: str = "f5tts") -> Optional[Dict[str, str]]:
        """
        Get voice information for a specific character.
        
        Args:
            character_name: Name of the character
            engine_type: "audio_only" (ChatterBox, VibeVoice, Higgs) or "audio_and_text" (F5-TTS) - determines if text is required
            
        Returns:
            Dict with voice info or None if not found
        """
        if not self._character_cache_valid:
            self._refresh_character_cache()
        
        # CRITICAL FIX: Resolve character aliases first before cache lookup
        resolved_character = self.resolve_character_alias(character_name)
        character_info = self._character_cache.get(resolved_character.lower())
        if not character_info:
            return None
        
        # For audio-only engines (ChatterBox, VibeVoice, Higgs), we don't require text files
        if engine_type == "audio_only" or engine_type == "chatterbox":  # Keep backward compatibility
            return {
                'audio_path': character_info['audio_path'],
                'character_name': character_name,
                'source_folder': character_info['source_folder']
            }
        
        # For audio+text engines (F5-TTS), we need both audio and text
        if engine_type == "audio_and_text" or engine_type == "f5tts":  # Keep backward compatibility
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
        
        # Scan models/TTS/voices/ directory for character folders
        models_tts_voices_dir = self._get_models_tts_voices_dir()
        if models_tts_voices_dir and os.path.exists(models_tts_voices_dir):
            self._scan_character_directories(models_tts_voices_dir, "models/TTS/voices")
        
        # Scan models/voices/ directory for character folders
        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir and os.path.exists(models_voices_dir):
            self._scan_character_directories(models_voices_dir, "models/voices")
        
        # Scan any additional extra_model_paths.yaml configured voices directories for characters
        if get_all_voices_paths:
            try:
                extra_voices_paths = get_all_voices_paths()
                # Keep track of already scanned paths to avoid duplicates
                scanned_paths = set()
                if models_voices_dir:
                    scanned_paths.add(os.path.normpath(models_voices_dir))
                models_tts_voices_dir = self._get_models_tts_voices_dir()
                if models_tts_voices_dir:
                    scanned_paths.add(os.path.normpath(models_tts_voices_dir))

                for extra_path in extra_voices_paths:
                    # Skip paths we already scanned above (normalize paths for comparison)
                    normalized_extra = os.path.normpath(extra_path)
                    if normalized_extra in scanned_paths:
                        continue
                    if os.path.exists(extra_path):
                        self._scan_character_directories(extra_path, f"extra:{extra_path}")
                        scanned_paths.add(normalized_extra)
            except Exception:
                # Silently handle any extra paths errors
                pass
        
        self._character_cache_valid = True
    
    def _refresh_character_aliases(self):
        """Refresh character aliases from alias map files."""
        self._character_aliases.clear()
        self._character_language_defaults.clear()
        
        # Load aliases in priority order (later loads override earlier ones)
        
        # 1. Load from voices_examples folder first (lowest priority)
        voices_examples_dir = self._get_voices_examples_dir()
        if voices_examples_dir:
            self._load_alias_file(voices_examples_dir, "voices_examples")
        
        # 2. Load from models/TTS/voices folder (medium priority - overrides examples)
        models_tts_voices_dir = self._get_models_tts_voices_dir()
        if models_tts_voices_dir and os.path.exists(models_tts_voices_dir):
            self._load_alias_file(models_tts_voices_dir, "models/TTS/voices")
        
        # 3. Load from models/voices folder (highest priority - overrides all others)
        models_voices_dir = self._get_models_voices_dir()
        if models_voices_dir:
            self._load_alias_file(models_voices_dir, "models/voices")
        
        # 4. Load from any additional extra_model_paths.yaml configured voices directories
        # These have higher priority than models/voices since they're user-configured
        if get_all_voices_paths:
            try:
                extra_voices_paths = get_all_voices_paths()
                # Keep track of already processed paths to avoid duplicates
                processed_paths = set()
                if models_voices_dir:
                    processed_paths.add(os.path.normpath(models_voices_dir))
                models_tts_voices_dir = self._get_models_tts_voices_dir()
                if models_tts_voices_dir:
                    processed_paths.add(os.path.normpath(models_tts_voices_dir))

                for extra_path in extra_voices_paths:
                    # Skip paths we already processed above (normalize paths for comparison)
                    normalized_extra = os.path.normpath(extra_path)
                    if normalized_extra in processed_paths:
                        continue
                    if os.path.exists(extra_path):
                        self._load_alias_file(extra_path, f"extra:{extra_path}")
                        processed_paths.add(normalized_extra)
            except Exception:
                # Silently handle any extra paths errors
                pass
        
        self._aliases_valid = True
    
    def _load_alias_file(self, base_dir: str, source_name: str):
        """Load character aliases from a #character_alias_map.txt file (or fallback to .json)."""
        # Try new txt format first
        txt_alias_file = os.path.join(base_dir, "#character_alias_map.txt")
        if os.path.exists(txt_alias_file):
            self._load_txt_alias_file(txt_alias_file, source_name)
            return
        
        # Fallback to old JSON format for backward compatibility
        json_alias_file = os.path.join(base_dir, "character_alias_map.json")
        if os.path.exists(json_alias_file):
            self._load_json_alias_file(json_alias_file, source_name)
    
    def _load_txt_alias_file(self, alias_file: str, source_name: str):
        """Load character aliases from txt file with flexible format."""
        try:
            with open(alias_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            loaded_count = 0
            for line_num, line in enumerate(lines, 1):
                # Strip whitespace
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Try parsing the line
                alias, target, language = self._parse_alias_line(line, line_num, alias_file)
                if alias and target:
                    # Convert to lowercase for consistent matching
                    self._character_aliases[alias.lower()] = target.lower()
                    if language:
                        self._character_language_defaults[alias.lower()] = language.lower()
                    loaded_count += 1
            
            if loaded_count > 0:
                pass  # print(f"🗂️ Character Aliases: Loaded {loaded_count} aliases from {source_name}")
                
        except Exception as e:
            print(f"⚠️ Character Aliases: Error reading {alias_file}: {e}")
    
    def _parse_alias_line(self, line: str, line_num: int, alias_file: str) -> tuple[str, str, str]:
        """Parse a single alias line supporting both = and tab formats with optional language."""
        alias = None
        target = None
        language = None
        
        try:
            # Try splitting on = first
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    alias = parts[0].strip()
                    right_side = parts[1].strip()
                    
                    # Check if right side has language (comma-separated)
                    if ',' in right_side:
                        target_parts = right_side.split(',', 1)
                        target = target_parts[0].strip()
                        language = target_parts[1].strip()
                    else:
                        target = right_side
                        
            # Try splitting on tab(s)
            elif '\t' in line:
                parts = line.split('\t')
                # Filter out empty parts (handles multiple tabs)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    alias = parts[0]
                    target = parts[1]
                    # Check for third part (language)
                    if len(parts) >= 3:
                        language = parts[2]
            
            # Validate
            if not alias or not target:
                print(f"⚠️ Character Aliases: Invalid format on line {line_num} in {alias_file}: '{line}'")
                return None, None, None
            
            return alias, target, language
            
        except Exception as e:
            print(f"⚠️ Character Aliases: Error parsing line {line_num} in {alias_file}: {e}")
            return None, None, None
    
    def _load_json_alias_file(self, alias_file: str, source_name: str):
        """Load character aliases from JSON file (backward compatibility)."""
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
    
    def get_character_language_defaults(self) -> Dict[str, str]:
        """
        Get character language defaults from alias system.
        
        Returns:
            Dictionary of character -> default language mappings
        """
        if not self._aliases_valid:
            self._refresh_character_aliases()
        
        return self._character_language_defaults.copy()
    
    def get_character_default_language(self, character: str) -> Optional[str]:
        """
        Get default language for a specific character.
        
        Args:
            character: Character name
            
        Returns:
            Default language code or None if not set
        """
        if not self._aliases_valid:
            self._refresh_character_aliases()
        
        return self._character_language_defaults.get(character.lower())
    
    def _scan_character_directories(self, base_dir: str, source_name: str):
        """
        Scan for character voices using filename-only approach.
        Folders are used for organization only, not as character names.
        
        Supported structure:
           base_dir/
           ├── female_01.wav
           ├── female_01.reference.txt (for F5TTS)
           ├── subfolder/
           │   ├── male_01.wav
           │   └── male_01.reference.txt (for F5TTS)
           └── alice.wav
               └── alice.reference.txt (for F5TTS)
        
        Character name is always extracted from the audio filename.
        """
        try:
            # Recursively scan all directories for audio files
            self._scan_all_audio_files_recursive(base_dir, source_name)
                    
        except Exception as e:
            print(f"⚠️ Character Discovery: Error scanning {source_name}: {e}")
    
    def _scan_all_audio_files_recursive(self, base_dir: str, source_name: str):
        """
        Recursively scan all directories for audio files and use filename as character name.
        Folders are ignored as character identifiers - they're only for organization.
        """
        try:
            for root, dirs, files in os.walk(base_dir):
                # Filter to only audio files
                audio_files = self._filter_audio_files(files)
                
                for audio_file in audio_files:
                    audio_path = os.path.join(root, audio_file)
                    
                    # Extract character name from filename (remove extension)
                    character_name = Path(audio_file).stem.lower()
                    
                    # Skip if this character was already found (first occurrence wins)
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
                        'character_directory': root  # Store the actual directory containing the file
                    }
                    
        except Exception as e:
            print(f"⚠️ Character Discovery: Error scanning audio files in {source_name}: {e}")
    
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


# Language-aware convenience functions

def get_character_language_defaults() -> Dict[str, str]:
    """Convenience function to get character language defaults."""
    return voice_discovery.get_character_language_defaults()


def get_character_default_language(character: str) -> Optional[str]:
    """Convenience function to get default language for a character."""
    return voice_discovery.get_character_default_language(character)