"""
Voice Discovery Utility for F5-TTS Integration
Enhanced voice file discovery with dual folder support and smart text file priority
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
try:
    import folder_paths
except ImportError:
    folder_paths = None


class VoiceDiscovery:
    """
    Enhanced voice discovery system for F5-TTS nodes.
    
    Features:
    - Dual folder support: models/voices/ and voices_examples/
    - Recursive folder scanning with nested directory support
    - Smart text file priority: .reference.txt > .txt
    - Performance caching to avoid repeated filesystem scans
    - Backward compatibility with existing flat structure
    """
    
    def __init__(self):
        self._cache = {}
        self._cache_valid = False
    
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
        print(f"🎤 Voice Discovery: Found {len(self._cache)} voices across all locations")
    
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