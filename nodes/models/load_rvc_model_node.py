"""
Load RVC Model Node - Loads RVC voice models for voice conversion
Adapted from reference implementation for TTS Suite integration
"""

import os
import re
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
                "index_mode": (["auto", "none", "custom"], {
                    "default": "auto",
                    "tooltip": "How to choose the FAISS index. Auto finds the most likely matching index for the selected model, None disables index usage, and Custom enables manual index selection."
                }),
                "index_file": (rvc_indexes, {
                    "default": "",
                    "tooltip": "Custom FAISS index file (.index). Only used when index mode is Custom."
                }),
                "training_artifacts": ("TRAINING_ARTIFACTS", {
                    "tooltip": "Optional output from the unified training node. When connected, the trained RVC model paths are used directly."
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
    
    def load_rvc_model(self, model, index_mode="auto", index_file="", training_artifacts=None, auto_download=True):
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
            artifact_model_path = None
            artifact_index_path = None
            if isinstance(training_artifacts, dict):
                artifact_engine = str(training_artifacts.get("engine_type", "") or "").strip().lower()
                artifact_model = training_artifacts.get("rvc_model")
                if artifact_engine == "rvc" and isinstance(artifact_model, dict):
                    candidate_model_path = artifact_model.get("model_path")
                    candidate_index_path = artifact_model.get("index_path")
                    if candidate_model_path and os.path.exists(candidate_model_path):
                        artifact_model_path = candidate_model_path
                        if candidate_index_path and os.path.exists(candidate_index_path):
                            artifact_index_path = candidate_index_path

            print(f"🎵 Loading RVC Model: {model}")

            model_path = artifact_model_path or self._get_model_path(model, auto_download)
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"RVC model not found: {model}")

            index_path, resolved_index_mode, resolved_index_label = self._resolve_index_selection(
                model_path=model_path,
                index_mode=index_mode,
                index_file=index_file,
                artifact_index_path=artifact_index_path,
                auto_download=auto_download,
            )

            model_name = os.path.basename(model_path)
            rvc_model = {
                "model_path": model_path,
                "index_path": index_path,
                "model_name": model_name,
                "index_name": os.path.basename(index_path) if index_path else None,
                "index_mode": resolved_index_mode,
                "type": "rvc_model"
            }

            model_info = (
                f"RVC Model: {model_name} | "
                f"Index ({resolved_index_mode}): {resolved_index_label} | "
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

    @staticmethod
    def _tokenize_name(value: str):
        return [token for token in re.split(r"[^a-z0-9]+", str(value or "").lower()) if token]

    @classmethod
    def _iter_local_rvc_index_paths(cls):
        seen = set()
        search_dirs = []

        try:
            from utils.models.extra_paths import get_all_tts_model_paths

            for base_path in get_all_tts_model_paths('TTS'):
                search_dirs.extend([
                    os.path.join(base_path, "RVC", ".index"),
                    os.path.join(base_path, "RVC"),
                ])
        except Exception:
            pass

        if folder_paths:
            models_dir = folder_paths.models_dir
            search_dirs.extend([
                os.path.join(models_dir, "TTS", "RVC", ".index"),
                os.path.join(models_dir, "TTS", "RVC"),
                os.path.join(models_dir, "RVC", ".index"),
                os.path.join(models_dir, "RVC"),
            ])

        for search_dir in search_dirs:
            normalized_dir = os.path.normpath(search_dir)
            if normalized_dir in seen or not os.path.isdir(search_dir):
                continue
            seen.add(normalized_dir)
            for file in os.listdir(search_dir):
                if file.endswith(".index"):
                    yield os.path.join(search_dir, file)

    @classmethod
    def _score_index_candidate(cls, model_path: str, index_path: str):
        model_stem = os.path.splitext(os.path.basename(model_path))[0].lower()
        index_stem = os.path.splitext(os.path.basename(index_path))[0].lower()

        score = 0
        if index_stem == model_stem:
            score += 120
        if index_stem.startswith(model_stem):
            score += 80
        elif model_stem in index_stem:
            score += 50

        model_tokens = set(cls._tokenize_name(model_stem))
        index_tokens = set(cls._tokenize_name(index_stem))
        shared_tokens = model_tokens & index_tokens
        score += len(shared_tokens) * 12

        for sample_rate_hint in ("32k", "40k", "48k"):
            if sample_rate_hint in model_tokens and sample_rate_hint in index_tokens:
                score += 18

        model_dir = os.path.dirname(model_path)
        index_dir = os.path.dirname(index_path)
        if index_dir == os.path.join(model_dir, ".index"):
            score += 24
        elif index_dir == model_dir:
            score += 16

        return score

    @classmethod
    def _auto_detect_index_path(cls, model_path: str):
        candidate_paths = []
        model_dir = os.path.dirname(model_path)
        candidate_paths.extend([
            os.path.join(model_dir, ".index"),
            model_dir,
        ])

        seen_paths = set()
        local_candidates = []
        for search_dir in candidate_paths:
            normalized_dir = os.path.normpath(search_dir)
            if normalized_dir in seen_paths or not os.path.isdir(search_dir):
                continue
            seen_paths.add(normalized_dir)
            for file in os.listdir(search_dir):
                if file.endswith(".index"):
                    local_candidates.append(os.path.join(search_dir, file))

        for index_path in cls._iter_local_rvc_index_paths():
            normalized_path = os.path.normpath(index_path)
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)
            local_candidates.append(index_path)

        scored_candidates = [
            (cls._score_index_candidate(model_path, index_path), index_path)
            for index_path in local_candidates
        ]
        scored_candidates = [
            (score, index_path)
            for score, index_path in scored_candidates
            if score > 0
        ]
        if not scored_candidates:
            return None

        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_path = scored_candidates[0]
        if best_score < 20:
            return None
        return best_path

    def _resolve_index_selection(self, model_path, index_mode, index_file, artifact_index_path=None, auto_download=True):
        mode = str(index_mode or "auto").strip().lower()
        if mode not in {"auto", "none", "custom"}:
            mode = "auto"

        if mode == "none":
            return None, "none", "None"

        if mode == "custom":
            if not index_file:
                return None, "custom", "None"
            index_path = self._get_index_path(index_file, auto_download)
            if index_path and os.path.exists(index_path):
                return index_path, "custom", os.path.basename(index_path)
            print(f"⚠️ Custom index not found: {index_file}, continuing without index")
            return None, "custom", "None"

        if artifact_index_path and os.path.exists(artifact_index_path):
            return artifact_index_path, "auto", os.path.basename(artifact_index_path)

        auto_index_path = self._auto_detect_index_path(model_path)
        if auto_index_path and os.path.exists(auto_index_path):
            return auto_index_path, "auto", os.path.basename(auto_index_path)
        return None, "auto", "None"
    
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
        
        # Add local models (respects extra_model_paths.yaml)
        try:
            from utils.models.extra_paths import get_all_tts_model_paths

            # Search in all configured TTS model paths
            all_tts_paths = get_all_tts_model_paths('TTS')
            for base_path in all_tts_paths:
                rvc_dir = os.path.join(base_path, "RVC")
                if os.path.exists(rvc_dir):
                    for file in os.listdir(rvc_dir):
                        if file.endswith('.pth'):
                            local_model = f"local:{file}"
                            if local_model not in models:
                                models.append(local_model)
        except Exception:
            pass

        # Fallback to hardcoded ComfyUI models_dir (legacy)
        try:
            if folder_paths:
                models_dir = folder_paths.models_dir
                rvc_search_paths = [
                    os.path.join(models_dir, "TTS", "RVC"),
                    os.path.join(models_dir, "RVC")  # Legacy
                ]

                for rvc_models_dir in rvc_search_paths:
                    if os.path.exists(rvc_models_dir):
                        for file in os.listdir(rvc_models_dir):
                            if file.endswith('.pth'):
                                local_model = f"local:{file}"
                                if local_model not in models:
                                    models.append(local_model)
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
        
        # Add local index files (respects extra_model_paths.yaml)
        try:
            from utils.models.extra_paths import get_all_tts_model_paths

            # Search in all configured TTS model paths
            all_tts_paths = get_all_tts_model_paths('TTS')
            for base_path in all_tts_paths:
                # Check both RVC/.index and RVC/ for index files
                index_search_paths = [
                    os.path.join(base_path, "RVC", ".index"),
                    os.path.join(base_path, "RVC")
                ]
                for rvc_index_dir in index_search_paths:
                    if os.path.exists(rvc_index_dir):
                        for file in os.listdir(rvc_index_dir):
                            if file.endswith('.index'):
                                local_index = f"local:{file}"
                                if local_index not in indexes:
                                    indexes.append(local_index)
        except Exception:
            pass

        # Fallback to hardcoded ComfyUI models_dir (legacy)
        try:
            if folder_paths:
                models_dir = folder_paths.models_dir
                index_search_paths = [
                    os.path.join(models_dir, "TTS", "RVC", ".index"),
                    os.path.join(models_dir, "RVC", ".index"),  # Legacy
                    os.path.join(models_dir, "TTS", "RVC"),
                    os.path.join(models_dir, "RVC")
                ]

                for rvc_index_dir in index_search_paths:
                    if os.path.exists(rvc_index_dir):
                        for file in os.listdir(rvc_index_dir):
                            if file.endswith('.index'):
                                local_index = f"local:{file}"
                                if local_index not in indexes:
                                    indexes.append(local_index)
        except:
            pass
        
        return sorted(indexes)
    
    def _get_model_path(self, model_name, auto_download=True):
        """Get full path to RVC model file (respects extra_model_paths.yaml)."""
        try:
            # Handle local: prefix (like F5-TTS pattern)
            if model_name.startswith("local:"):
                actual_model_name = model_name.replace("local:", "")

                # Search in extra_model_paths.yaml first
                try:
                    from utils.models.extra_paths import get_all_tts_model_paths
                    all_tts_paths = get_all_tts_model_paths('TTS')

                    for base_path in all_tts_paths:
                        model_path = os.path.join(base_path, "RVC", actual_model_name)
                        if os.path.exists(model_path):
                            return model_path
                except Exception:
                    pass

                # Fallback to hardcoded ComfyUI paths
                if folder_paths:
                    models_dir = folder_paths.models_dir
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
        """Get full path to RVC index file (respects extra_model_paths.yaml)."""
        try:
            # Handle local: prefix (like F5-TTS pattern)
            if index_name.startswith("local:"):
                actual_index_name = index_name.replace("local:", "")

                # Search in extra_model_paths.yaml first
                try:
                    from utils.models.extra_paths import get_all_tts_model_paths
                    all_tts_paths = get_all_tts_model_paths('TTS')

                    for base_path in all_tts_paths:
                        # Check both .index subdirectory and RVC root
                        index_paths = [
                            os.path.join(base_path, "RVC", ".index", actual_index_name),
                            os.path.join(base_path, "RVC", actual_index_name)
                        ]
                        for index_path in index_paths:
                            if os.path.exists(index_path):
                                return index_path
                except Exception:
                    pass

                # Fallback to hardcoded ComfyUI paths
                if folder_paths:
                    models_dir = folder_paths.models_dir
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
