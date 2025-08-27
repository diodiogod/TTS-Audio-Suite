"""
Load RVC Model Node - Loads RVC voice models for voice conversion
Adapted from reference implementation for TTS Suite integration
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional

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

# Import ComfyUI folder paths
try:
    import folder_paths
except ImportError:
    # Fallback for testing
    folder_paths = None


class LoadRVCModelNode(BaseTTSNode):
    """
    Load RVC Model Node - Loads trained RVC voice models.
    
    Loads RVC .pth models and optional FAISS index files for voice conversion.
    Output connects to narrator_target input on Voice Changer node.
    """
    
    @classmethod
    def NAME(cls):
        return "🎭 Load RVC Character Model"
    
    @classmethod  
    def INPUT_TYPES(cls):
        # Get available RVC models
        rvc_models = cls._get_available_rvc_models()
        rvc_indexes = cls._get_available_rvc_indexes()
        
        return {
            "required": {
                "model": (rvc_models, {
                    "default": rvc_models[0] if rvc_models else "Claire.pth",
                    "tooltip": "RVC trained voice model (.pth file). This determines the target voice characteristics."
                })
            },
            "optional": {
                "index_file": (rvc_indexes, {
                    "default": "",
                    "tooltip": "FAISS index file (.index) - Optional enhancement for better voice similarity. Leave empty if you don't have one."
                }),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically download model if not found locally"
                })
            }
        }
    
    RETURN_TYPES = ("RVC_MODEL", "STRING")
    RETURN_NAMES = ("rvc_model", "model_info")
    
    CATEGORY = "TTS Audio Suite/🎭 Voice & Character"
    
    FUNCTION = "load_rvc_model"
    
    DESCRIPTION = """
    Load RVC Model - Load trained RVC voice models for conversion
    
    Loads RVC (Real-time Voice Conversion) models trained on specific voices.
    These models learn the vocal characteristics of a target speaker.
    
    Key Features:
    • Supports .pth RVC model files
    • Optional FAISS index files for better similarity
    • Auto-download missing models
    • Model validation and caching
    
    Usage:
    • Connect output to narrator_target on Voice Changer node
    • Models should be placed in ComfyUI/models/RVC/ folder  
    • Index files should be in ComfyUI/models/RVC/.index/ folder
    
    Model Guide:
    • RVC models are speaker-specific (one model per voice)
    • Higher quality models require more training data
    • FAISS index improves voice similarity but increases processing time
    """
    
    def load_rvc_model(self, model, index_file="", auto_download=True):
        """
        Load RVC model and optional index file.
        
        Args:
            model: RVC model filename (.pth)
            index_file: Optional FAISS index filename (.index)
            auto_download: Whether to auto-download missing models
            
        Returns:
            Tuple of (rvc_model_dict, model_info)
        """
        try:
            print(f"🎵 Loading RVC Model: {model}")
            
            # Get model path
            model_path = self._get_model_path(model, auto_download)
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"RVC model not found: {model}")
            
            # Get index path if specified
            index_path = None
            if index_file:
                index_path = self._get_index_path(index_file, auto_download)
                if index_path and not os.path.exists(index_path):
                    print(f"⚠️ Index file not found: {index_file}, continuing without index")
                    index_path = None
            
            # Create RVC model configuration
            rvc_model = {
                "model_path": model_path,
                "index_path": index_path,
                "model_name": os.path.basename(model),
                "index_name": os.path.basename(index_file) if index_file else None,
                "type": "rvc_model"
            }
            
            # Create model info
            model_info = (
                f"RVC Model: {os.path.basename(model)} | "
                f"Index: {os.path.basename(index_file) if index_file else 'None'} | "
                f"Path: {model_path}"
            )
            
            print(f"✅ RVC model loaded successfully")
            return rvc_model, model_info
            
        except Exception as e:
            print(f"❌ Failed to load RVC model: {e}")
            # Return empty model on error
            empty_model = {
                "model_path": None,
                "index_path": None, 
                "model_name": None,
                "index_name": None,
                "type": "rvc_model"
            }
            error_info = f"RVC Model Load Error: {str(e)}"
            return empty_model, error_info
    
    @classmethod
    def _get_available_rvc_models(cls):
        """Get list of available RVC model files."""
        # Start with downloadable models (like F5-TTS pattern)
        try:
            from utils.downloads.model_downloader import AVAILABLE_RVC_MODELS
            # Extract just the model names from the full paths
            models = [os.path.basename(model_path) for model_path in AVAILABLE_RVC_MODELS]
        except ImportError:
            # Fallback if downloader not available
            models = [
                "Claire.pth",
                "Sayano.pth", 
                "Mae_v2.pth",
                "Fuji.pth",
                "Monika.pth"
            ]
        
        # Add local models (like F5-TTS pattern)
        try:
            if folder_paths:
                models_dir = folder_paths.models_dir
                # Try TTS path first, then legacy
                rvc_search_paths = [
                    os.path.join(models_dir, "TTS", "RVC"),
                    os.path.join(models_dir, "RVC")  # Legacy
                ]
                
                for rvc_models_dir in rvc_search_paths:
                    if os.path.exists(rvc_models_dir):
                        for file in os.listdir(rvc_models_dir):
                            if file.endswith('.pth') and f"local:{file}" not in models:
                                # Add local: prefix to distinguish from downloadable ones
                                models.append(f"local:{file}")
        except:
            pass
        
        return sorted(models)
    
    @classmethod
    def _get_available_rvc_indexes(cls):
        """Get list of available RVC index files."""
        indexes = [""]  # Empty option first
        
        # Add downloadable index files (like F5-TTS pattern)
        try:
            from utils.downloads.model_downloader import AVAILABLE_RVC_INDEXES
            # Extract just the index names from the full paths
            for index_path in AVAILABLE_RVC_INDEXES:
                index_name = os.path.basename(index_path)
                indexes.append(index_name)
        except ImportError:
            # Fallback if downloader not available
            indexes.extend([
                "added_IVF1063_Flat_nprobe_1_Sayano_v2.index",
                "added_IVF985_Flat_nprobe_1_Fuji_v2.index", 
                "Monika_v2_40k.index",
                "Sayano_v2_40k.index"
            ])
        
        # Add local index files (like F5-TTS pattern)
        try:
            if folder_paths:
                models_dir = folder_paths.models_dir
                # Try TTS path first, then legacy
                index_search_paths = [
                    os.path.join(models_dir, "TTS", "RVC", ".index"),
                    os.path.join(models_dir, "RVC", ".index")  # Legacy
                ]
                
                for rvc_index_dir in index_search_paths:
                    if os.path.exists(rvc_index_dir):
                        for file in os.listdir(rvc_index_dir):
                            if file.endswith('.index') and f"local:{file}" not in indexes:
                                # Add local: prefix to distinguish from downloadable ones
                                indexes.append(f"local:{file}")
        except:
            pass
        
        return sorted(indexes)
    
    def _get_model_path(self, model_name, auto_download=True):
        """Get full path to RVC model file."""
        try:
            # Handle local: prefix (like F5-TTS pattern)
            if model_name.startswith("local:"):
                actual_model_name = model_name.replace("local:", "")
                if folder_paths:
                    models_dir = folder_paths.models_dir
                    # Try TTS path first, then legacy
                    search_paths = [
                        os.path.join(models_dir, "TTS", "RVC", actual_model_name),
                        os.path.join(models_dir, "RVC", actual_model_name)  # Legacy
                    ]
                    
                    for model_path in search_paths:
                        if os.path.exists(model_path):
                            return model_path
                return None
            
            # Regular downloadable model
            if folder_paths:
                models_dir = folder_paths.models_dir
                # Try TTS path first, then legacy
                tts_path = os.path.join(models_dir, "TTS", "RVC", model_name)
                legacy_path = os.path.join(models_dir, "RVC", model_name)
                
                if os.path.exists(tts_path):
                    return tts_path
                elif os.path.exists(legacy_path):
                    return legacy_path
                    
                # Auto-download if enabled - download to TTS path
                if auto_download:
                    downloaded_path = self._download_rvc_model(model_name, tts_path)
                    if downloaded_path:
                        return downloaded_path
            
            return None
        except Exception as e:
            print(f"Error getting model path: {e}")
            return None
    
    def _get_index_path(self, index_name, auto_download=True):
        """Get full path to RVC index file."""
        try:
            # Handle local: prefix (like F5-TTS pattern)
            if index_name.startswith("local:"):
                actual_index_name = index_name.replace("local:", "")
                if folder_paths:
                    models_dir = folder_paths.models_dir
                    # Try TTS path first, then legacy
                    search_paths = [
                        os.path.join(models_dir, "TTS", "RVC", ".index", actual_index_name),
                        os.path.join(models_dir, "RVC", ".index", actual_index_name)  # Legacy
                    ]
                    
                    for index_path in search_paths:
                        if os.path.exists(index_path):
                            return index_path
                return None
            
            # Regular downloadable index
            if folder_paths:
                models_dir = folder_paths.models_dir
                # Try TTS path first, then legacy
                tts_path = os.path.join(models_dir, "TTS", "RVC", ".index", index_name)
                legacy_path = os.path.join(models_dir, "RVC", ".index", index_name)
                
                if os.path.exists(tts_path):
                    return tts_path
                elif os.path.exists(legacy_path):
                    return legacy_path
                    
                # Auto-download if enabled - download to TTS path
                if auto_download:
                    downloaded_path = self._download_rvc_index(index_name, tts_path)
                    if downloaded_path:
                        return downloaded_path
            
            return None
        except Exception as e:
            print(f"Error getting index path: {e}")
            return None
    
    def _download_rvc_model(self, model_name, target_path):
        """Download RVC model if not available locally."""
        try:
            from utils.downloads.model_downloader import download_rvc_model
            
            print(f"📥 Attempting to auto-download RVC model: {model_name}")
            downloaded_path = download_rvc_model(model_name)
            
            if downloaded_path and os.path.exists(downloaded_path):
                return downloaded_path
            else:
                print(f"❌ Auto-download failed for {model_name}")
                return None
                
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    def _download_rvc_index(self, index_name, target_path):
        """Download RVC index if not available locally."""
        try:
            from utils.downloads.model_downloader import download_rvc_index
            
            print(f"📥 Attempting to auto-download RVC index: {index_name}")
            downloaded_path = download_rvc_index(index_name)
            
            if downloaded_path and os.path.exists(downloaded_path):
                return downloaded_path
            else:
                print(f"❌ Auto-download failed for {index_name}")
                return None
                
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs for RVC model loading."""
        return True