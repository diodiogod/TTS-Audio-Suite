"""
Extra Model Paths Support for TTS Audio Suite
Implements support for ComfyUI's extra_model_paths.yaml configuration file
"""

import os
import yaml
import folder_paths
from typing import List, Dict, Any, Optional
import logging

class TtsExtraPathsManager:
    """
    Manager for TTS model paths that respects ComfyUI's extra_model_paths.yaml configuration.
    
    This ensures TTS Audio Suite uses the same shared model directories as other ComfyUI nodes,
    preventing model duplication and enabling shared storage configurations.
    """
    
    def __init__(self):
        self.tts_folders = {}
        self._load_tts_paths()
    
    def _load_tts_paths(self):
        """Load TTS-specific paths from ComfyUI's folder_paths system"""
        # First, check if TTS folders are already registered with ComfyUI
        if hasattr(folder_paths, 'folder_names_and_paths'):
            global_folders = folder_paths.folder_names_and_paths
            
            # Look for existing TTS-related folders
            for folder_type in ['TTS', 'tts', 'text_to_speech', 'voice_models']:
                if folder_type in global_folders:
                    self.tts_folders[folder_type] = global_folders[folder_type][0]  # Get paths list
        
        # If no TTS folders registered, create default TTS structure
        if not self.tts_folders:
            self._setup_default_tts_paths()
    
    def _setup_default_tts_paths(self):
        """Set up default TTS paths using ComfyUI's models directory structure"""
        models_dir = folder_paths.models_dir
        
        # Register TTS folder types with ComfyUI's folder_paths system
        tts_model_types = {
            'TTS': {
                'paths': [os.path.join(models_dir, 'TTS')],
                'extensions': {'.safetensors', '.bin', '.pt', '.pth', '.ckpt', 'folder'}
            },
            'voices': {
                'paths': [
                    os.path.join(models_dir, 'voices'),           # Primary: standard ComfyUI location
                    os.path.join(models_dir, 'TTS', 'voices')     # Fallback: logical TTS organization
                ],
                'extensions': {'.wav', '.mp3', '.flac', '.ogg', 'folder'}
            }
        }
        
        # Register with ComfyUI's system so other nodes can also use these paths
        for folder_type, config in tts_model_types.items():
            folder_paths.add_model_folder_path(folder_type, config['paths'][0])
            self.tts_folders[folder_type] = config['paths']
    
    def get_tts_model_directory(self, model_type: str = 'TTS') -> str:
        """
        Get the primary TTS model directory for a given type.
        
        Args:
            model_type: Type of TTS model ('TTS', 'TTS_voices', etc.)
            
        Returns:
            Primary directory path for this model type
        """
        if model_type in self.tts_folders and self.tts_folders[model_type]:
            return self.tts_folders[model_type][0]  # Return first (primary) path
        
        # Fallback to default if not configured
        models_dir = folder_paths.models_dir
        return os.path.join(models_dir, 'TTS')
    
    def get_all_tts_model_paths(self, model_type: str = 'TTS') -> List[str]:
        """
        Get all configured TTS model directories for a given type.
        
        This includes both the default ComfyUI models folder and any extra_model_paths
        configured directories.
        
        Args:
            model_type: Type of TTS model ('TTS', 'TTS_voices', etc.)
            
        Returns:
            List of all directory paths for this model type
        """
        if model_type in self.tts_folders:
            return self.tts_folders[model_type].copy()
        
        # Fallback to default
        models_dir = folder_paths.models_dir
        return [os.path.join(models_dir, 'TTS')]
    
    def find_model_in_paths(self, model_name: str, model_type: str = 'TTS', 
                           subdirs: Optional[List[str]] = None) -> Optional[str]:
        """
        Find a model file or directory in any of the configured TTS paths.
        
        Args:
            model_name: Name of the model to find
            model_type: Type of TTS model ('TTS', 'TTS_voices', etc.)
            subdirs: Optional list of subdirectories to check (e.g., ['chatterbox', 'f5tts'])
            
        Returns:
            Full path to the model if found, None otherwise
        """
        search_paths = self.get_all_tts_model_paths(model_type)
        
        # Search in all configured paths
        for base_path in search_paths:
            # Search directly in base path
            if self._check_model_at_path(base_path, model_name):
                return os.path.join(base_path, model_name)
            
            # Search in subdirectories if specified
            if subdirs:
                for subdir in subdirs:
                    subpath = os.path.join(base_path, subdir)
                    if self._check_model_at_path(subpath, model_name):
                        return os.path.join(subpath, model_name)
        
        return None
    
    def _check_model_at_path(self, base_path: str, model_name: str) -> bool:
        """Check if a model exists at the given path"""
        if not os.path.exists(base_path):
            return False
        
        full_path = os.path.join(base_path, model_name)
        
        # Check if it's a file
        if os.path.isfile(full_path):
            return True
        
        # Check if it's a directory (for model folders)
        if os.path.isdir(full_path):
            return True
        
        return False
    
    def get_preferred_download_path(self, model_type: str = 'TTS', 
                                  engine_name: Optional[str] = None) -> str:
        """
        Get the preferred path for downloading new models.
        
        This respects extra_model_paths.yaml configuration - if a user has configured
        a shared models directory, new downloads will go there instead of the default.
        
        Args:
            model_type: Type of TTS model ('TTS', 'TTS_voices', etc.)
            engine_name: Optional engine name for organization (e.g., 'chatterbox', 'f5tts')
            
        Returns:
            Full path where new models should be downloaded
        """
        # Get the primary (first) configured path
        base_path = self.get_tts_model_directory(model_type)
        
        # Add engine subdirectory if specified
        if engine_name:
            base_path = os.path.join(base_path, engine_name)
        
        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)
        
        return base_path
    
    def register_tts_engine_paths(self, engine_name: str, custom_paths: Dict[str, str]):
        """
        Register custom paths for a specific TTS engine.
        
        Args:
            engine_name: Name of the engine (e.g., 'chatterbox', 'f5tts')
            custom_paths: Dictionary of path types to paths
        """
        for path_type, path in custom_paths.items():
            folder_type = f"TTS_{engine_name}_{path_type}"
            folder_paths.add_model_folder_path(folder_type, path)
            self.tts_folders[folder_type] = [path]
    
    def get_voices_directory(self) -> str:
        """Get the primary directory for voice files, respecting extra_model_paths.yaml"""
        # Check for configured voices paths first
        if 'voices' in self.tts_folders and self.tts_folders['voices']:
            primary_path = self.tts_folders['voices'][0]
            os.makedirs(primary_path, exist_ok=True)
            return primary_path
        
        # Fallback to default models/voices
        models_dir = folder_paths.models_dir
        voices_dir = os.path.join(models_dir, 'voices')
        os.makedirs(voices_dir, exist_ok=True)
        return voices_dir
    
    def get_all_voices_paths(self) -> List[str]:
        """Get all configured voice directories including fallbacks"""
        paths = []
        
        # Add configured voices paths (includes both models/voices and models/TTS/voices)
        if 'voices' in self.tts_folders:
            paths.extend(self.tts_folders['voices'])
        else:
            # Fallback to default structure if not configured
            models_dir = folder_paths.models_dir
            paths.extend([
                os.path.join(models_dir, 'voices'),           # Primary
                os.path.join(models_dir, 'TTS', 'voices')     # Fallback
            ])
        
        # Note: voices_examples/ is handled by the existing VoiceDiscovery class
        # We don't include it here to avoid duplicating that functionality
        
        return paths


# Global instance
_tts_paths_manager = TtsExtraPathsManager()

def get_tts_model_directory(model_type: str = 'TTS') -> str:
    """Get the primary TTS model directory for downloads"""
    return _tts_paths_manager.get_tts_model_directory(model_type)

def get_all_tts_model_paths(model_type: str = 'TTS') -> List[str]:
    """Get all TTS model search paths"""
    return _tts_paths_manager.get_all_tts_model_paths(model_type)

def find_model_in_paths(model_name: str, model_type: str = 'TTS', 
                       subdirs: Optional[List[str]] = None) -> Optional[str]:
    """Find a model in any configured TTS path"""
    return _tts_paths_manager.find_model_in_paths(model_name, model_type, subdirs)

def get_preferred_download_path(model_type: str = 'TTS', 
                               engine_name: Optional[str] = None) -> str:
    """Get preferred path for new model downloads"""
    return _tts_paths_manager.get_preferred_download_path(model_type, engine_name)

def get_voices_directory() -> str:
    """Get the voices directory respecting extra_model_paths.yaml"""
    return _tts_paths_manager.get_voices_directory()

def get_all_voices_paths() -> List[str]:
    """Get all configured voice search paths"""
    return _tts_paths_manager.get_all_voices_paths()

def register_tts_engine_paths(engine_name: str, custom_paths: Dict[str, str]):
    """Register custom paths for a specific TTS engine"""
    return _tts_paths_manager.register_tts_engine_paths(engine_name, custom_paths)

# Initialize TTS paths with ComfyUI integration
def initialize_tts_paths():
    """Initialize TTS paths integration with ComfyUI's folder_paths system"""
    global _tts_paths_manager
    _tts_paths_manager._load_tts_paths()

# Auto-initialize on import
initialize_tts_paths()