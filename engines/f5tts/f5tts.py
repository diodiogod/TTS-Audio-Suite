"""
ChatterBox F5-TTS Wrapper
Bridges F5-TTS API with ChatterBox interface standards
"""

import os
import sys
import torch
import tempfile
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import folder_paths

def detect_tokenizer_files(repo_id: str, base_path: str = "") -> Tuple[Optional[str], Optional[str]]:
    """
    Detect available tokenizer files (vocab.txt and/or tokenizer.json) in a HuggingFace repo.

    Args:
        repo_id: HuggingFace repository ID
        base_path: Base path within the repo (e.g., "F5TTS_Base/")

    Returns:
        Tuple of (vocab_filename, tokenizer_filename) where available, None otherwise
    """
    from huggingface_hub import hf_api

    vocab_filename = None
    tokenizer_filename = None

    try:
        api = hf_api.HfApi()
        repo_files = api.list_repo_files(repo_id)

        # Look for vocab.txt and tokenizer.json in the specified path
        for file_path in repo_files:
            if file_path.startswith(base_path):
                filename = file_path[len(base_path):] if base_path else file_path

                if filename == "vocab.txt" or (filename.endswith(".txt") and "vocab" in filename.lower()):
                    vocab_filename = file_path
                elif filename == "tokenizer.json" or (filename.endswith(".json") and "tokenizer" in filename.lower()):
                    tokenizer_filename = file_path

    except Exception as e:
        print(f"⚠️ Could not detect tokenizer files for {repo_id}: {e}")

    return vocab_filename, tokenizer_filename

# F5-TTS sample rate constant
F5TTS_SAMPLE_RATE = 24000

# F5-TTS model configurations
F5TTS_MODELS = {
    "F5TTS_Base": {"repo": "SWivid/F5-TTS", "exp": "F5TTS_Base", "step": 1200000, "ext": "safetensors"},
    "F5TTS_v1_Base": {"repo": "SWivid/F5-TTS", "exp": "F5TTS_v1_Base", "step": 1250000, "ext": "safetensors"},
    "E2TTS_Base": {"repo": "SWivid/E2-TTS", "exp": "E2TTS_Base", "step": 1200000, "ext": "safetensors"},
    "F5-DE": {"repo": "aihpi/F5-TTS-German", "exp": "F5TTS_Base", "step": 365000, "ext": "safetensors"},
    "F5-ES": {"repo": "jpgallegoar/F5-Spanish", "exp": "", "step": 1200000, "ext": "safetensors"},
    "F5-FR": {"repo": "RASPIAUDIO/F5-French-MixedSpeakers-reduced", "exp": "", "step": 1374000, "ext": "pt"},
    "F5-JP": {"repo": "Jmica/F5TTS", "exp": "JA_21999120", "step": 21999120, "ext": "pt"},
    "F5-IT": {"repo": "alien79/F5-TTS-italian", "exp": "", "step": 159600, "ext": "safetensors"},
    "F5-TH": {"repo": "VIZINTZOR/F5-TTS-THAI", "exp": "", "step": 1000000, "ext": "pt"},
    "F5-PT-BR": {"repo": "firstpixel/F5-TTS-pt-br", "exp": "pt-br", "step": 200000, "ext": "pt"},
    "F5-Polish": {"repo": "Gregniuki/F5-tts_English_German_Polish", "exp": "multi3", "step": 900000, "ext": "pt", "note": "High quality multilingual Polish/English/German model from reference implementation"},
    "F5-Hindi-Small": {"repo": "SPRINGLab/F5-Hindi-24KHz", "exp": "", "step": 2500000, "ext": "safetensors", "note": "Hindi Small model (151M params) from IIT Madras"},
}

def get_f5tts_models():
    """Get list of available F5-TTS models"""
    models = list(F5TTS_MODELS.keys())
    
    # Check for local models in ComfyUI models directory
    # Primary location: models/F5-TTS/
    # Search paths: TTS/F5-TTS first, then legacy paths for compatibility
    search_paths = [
        os.path.join(folder_paths.models_dir, "TTS", "F5-TTS"),
        os.path.join(folder_paths.models_dir, "F5-TTS"),  # Legacy
        os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS")  # Legacy fallback
    ]
    
    for models_dir in search_paths:
        try:
            if os.path.exists(models_dir):
                for item in os.listdir(models_dir):
                    item_path = os.path.join(models_dir, item)
                    if os.path.isdir(item_path):
                        # Check if it contains model files
                        for ext in [".safetensors", ".pt"]:
                            if any(f.endswith(ext) for f in os.listdir(item_path)):
                                local_model = f"local:{item}"
                                if local_model not in models:
                                    models.append(local_model)
                                break
        except Exception:
            pass  # Ignore errors in model discovery
    
    return models


class ChatterBoxF5TTS:
    """
    F5-TTS wrapper class following ChatterBox patterns.
    Bridges F5-TTS API with ChatterBox interface standards.
    """
    
    def __init__(self, model_name: str, device: str, ckpt_dir: Optional[str] = None):
        """Initialize F5-TTS model similar to ChatterboxTTS pattern"""
        self.sr = F5TTS_SAMPLE_RATE
        self.device = device
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.f5tts_model = None
        self.vocoder = None
        self.mel_spec_type = "vocos"  # Default vocoder
        
        
        # Initialize F5-TTS
        self._load_f5tts()

    def _download_tokenizer_file(self, repo_id: str, filename: str, file_type: str) -> str:
        """
        Download vocab.txt or tokenizer.json file for a model.

        Args:
            repo_id: HuggingFace repository ID
            filename: File name to download (e.g., "vocab.txt", "tokenizer.json")
            file_type: Type for logging ("vocab" or "tokenizer")

        Returns:
            Path to downloaded file
        """
        from huggingface_hub import hf_hub_download

        local_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", self.model_name, os.path.basename(filename))

        if os.path.exists(local_path):
            print(f"📁 Using local {file_type}: {local_path}")
            return local_path

        # Check HuggingFace cache first
        try:
            hf_cached_file = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            print(f"📁 Using cached {file_type}: {hf_cached_file}")
            return hf_cached_file
        except Exception:
            pass

        # Download to local directory
        print(f"📥 Downloading {file_type} to local directory: {filename}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            # Use unified downloader
            from utils.downloads.unified_downloader import unified_downloader

            files_to_download = [{
                'remote': filename,
                'local': os.path.basename(filename)
            }]

            downloaded_dir = unified_downloader.download_huggingface_model(
                repo_id=repo_id,
                model_name=self.model_name,
                files=files_to_download,
                engine_type="F5-TTS"
            )

            if downloaded_dir:
                return os.path.join(downloaded_dir, os.path.basename(filename))
            else:
                raise Exception(f"Unified {file_type} download failed")

        except Exception as e:
            print(f"⚠️ Failed to download {file_type} to local directory, using HF cache: {e}")
            return hf_hub_download(repo_id=repo_id, filename=filename)

    def _setup_vocos_redirect(self, vocos_dir: str):
        """
        Monkey patch HuggingFace downloads to use our organized Vocos location.
        This prevents F5-TTS from downloading Vocos to cache when it's already organized.
        """
        try:
            import huggingface_hub
            
            # Store original download function
            original_hf_hub_download = huggingface_hub.hf_hub_download
            
            def patched_hf_hub_download(repo_id, filename, **kwargs):
                # Intercept Vocos downloads and redirect to our organized location
                if repo_id == "charactr/vocos-mel-24khz":
                    local_file = os.path.join(vocos_dir, filename)
                    if os.path.exists(local_file):
                        print(f"📁 Using organized Vocos file: {local_file}")
                        return local_file
                
                # For all other downloads, use original function
                return original_hf_hub_download(repo_id, filename, **kwargs)
            
            # Apply the monkey patch
            huggingface_hub.hf_hub_download = patched_hf_hub_download
            # Redirecting Vocos downloads to TTS folder
            
        except Exception as e:
            print(f"⚠️ Failed to setup Vocos redirect: {e}")
            print("🔄 F5-TTS will download Vocos to cache as fallback")
    
    def _load_f5tts(self):
        """Load F5-TTS model and vocoder"""
        try:
            # Import os at the top to avoid scoping issues
            import os
            
            # Download and setup Vocos redirect BEFORE any F5-TTS operations
            from utils.downloads.unified_downloader import unified_downloader
            vocos_dir = unified_downloader.download_vocos_model()
            if vocos_dir:
                self._setup_vocos_redirect(vocos_dir)
            
            # Try to import F5-TTS
            from engines.f5_tts.api import F5TTS
            
            # Check if we have local model directory
            if self.ckpt_dir and os.path.exists(self.ckpt_dir):
                # Find model file and tokenizer files in local directory
                model_file = None
                vocab_file = None
                tokenizer_file = None
                for file in os.listdir(self.ckpt_dir):
                    if file.endswith((".safetensors", ".pt")):
                        model_file = os.path.join(self.ckpt_dir, file)
                    elif file.endswith(".txt") and "vocab" in file.lower():
                        vocab_file = os.path.join(self.ckpt_dir, file)
                    elif file.endswith(".json") and "tokenizer" in file.lower():
                        tokenizer_file = os.path.join(self.ckpt_dir, file)

                # Show detected files
                if tokenizer_file:
                    print(f"🔤 Detected tokenizer.json: {os.path.basename(tokenizer_file)}")
                if vocab_file:
                    print(f"📝 Detected vocab.txt: {os.path.basename(vocab_file)}")

                # Determine which tokenizer file to use (prefer tokenizer.json over vocab.txt)
                final_tokenizer_path = tokenizer_file if tokenizer_file else vocab_file
                
                if model_file:
                    # Load with explicit local files - determine correct config
                    model_config = "F5TTS_Base"  # Default config
                    if "v1" in self.model_name.lower() or "1.1" in self.model_name.lower():
                        model_config = "F5TTS_v1_Base"
                    elif "e2tts" in self.model_name.lower():
                        model_config = "E2TTS_Base"
                    elif ("small" in self.model_name.lower() or 
                          "155m" in self.model_name.lower()):
                        model_config = "F5TTS_Small"
                        print(f"🔍 Detected Small model architecture for local '{self.model_name}'")
                    # Language models use base configs
                    
                    # Get local Vocos path
                    vocos_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "vocos")
                    vocoder_local_path = vocos_path if os.path.exists(vocos_path) else None
                    
                    # Handle missing tokenizer files by providing explicit path to default vocab
                    if not final_tokenizer_path:
                        # Use the bundled vocab file from our F5-TTS installation
                        current_dir = os.path.dirname(__file__)
                        default_vocab = os.path.join(current_dir, "..", "f5_tts", "infer", "examples", "vocab.txt")
                        default_vocab = os.path.normpath(default_vocab)
                        if os.path.exists(default_vocab):
                            final_tokenizer_path = default_vocab
                        else:
                            final_tokenizer_path = ""  # Fall back to empty string
                    
                    self.f5tts_model = F5TTS(
                        model=model_config,
                        ckpt_file=model_file,
                        vocab_file=final_tokenizer_path,  # Pass tokenizer.json or vocab.txt
                        vocoder_local_path=vocoder_local_path,
                        device=self.device
                    )

                    # Store tokenizer information for text processing
                    self.tokenizer_file = tokenizer_file
                    self.vocab_file = vocab_file
                    # Store original model name for phonemizer on the actual model object
                    self.f5tts_model.original_model_name = self.model_name
                    if hasattr(self.f5tts_model, 'ema_model'):
                        self.f5tts_model.ema_model.original_model_name = self.model_name
                    print(f"📦 F5-TTS {model_config} loaded from local files")
                    return
            
            # Determine model configuration for HuggingFace models
            if self.model_name.startswith("local:"):
                # Local model
                local_name = self.model_name[6:]  # Remove "local:" prefix
                # Try TTS path first, then legacy paths
                search_paths = [
                    os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", local_name),
                    os.path.join(folder_paths.models_dir, "F5-TTS", local_name),  # Legacy
                    os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS", local_name)  # Legacy
                ]
                
                model_path = None
                for path in search_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                if not model_path:
                    raise ValueError(f"Local model not found: {local_name}")
                
                # Find model file
                model_file = None
                vocab_file = None
                for file in os.listdir(model_path):
                    if file.endswith((".safetensors", ".pt")):
                        model_file = os.path.join(model_path, file)
                    elif file.endswith(".txt") and "vocab" in file.lower():
                        vocab_file = os.path.join(model_path, file)
                    elif file.endswith(".json") and "tokenizer" in file.lower():
                        vocab_file = os.path.join(model_path, file)
                
                if not model_file:
                    raise FileNotFoundError(f"No model file found in {model_path}")
                
                # Load local model - determine correct config based on folder name
                model_config = "F5TTS_Base"  # Default config
                if "v1" in local_name.lower() or "1.1" in local_name.lower():
                    model_config = "F5TTS_v1_Base"
                elif "e2tts" in local_name.lower():
                    model_config = "E2TTS_Base"
                elif ("small" in local_name.lower() or 
                      "155m" in local_name.lower()):
                    model_config = "F5TTS_Small"
                    print(f"🔍 Detected Small model architecture for local folder '{local_name}'")
                # Language models use base configs - they don't have their own
                
                # Get local Vocos path
                vocos_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "vocos")
                vocoder_local_path = vocos_path if os.path.exists(vocos_path) else None
                
                # Handle missing vocab file by providing explicit path to default vocab
                if not vocab_file:
                    # Use the bundled vocab file from our F5-TTS installation
                    current_dir = os.path.dirname(__file__)
                    default_vocab = os.path.join(current_dir, "..", "f5_tts", "infer", "examples", "vocab.txt")
                    default_vocab = os.path.normpath(default_vocab)
                    if os.path.exists(default_vocab):
                        vocab_file = default_vocab
                    else:
                        vocab_file = ""  # Fall back to empty string
                
                self.f5tts_model = F5TTS(
                    model=model_config,
                    ckpt_file=model_file,
                    vocab_file=vocab_file,
                    vocoder_local_path=vocoder_local_path,
                    device=self.device
                )
                print(f"📦 F5-TTS {model_config} loaded from local files")
                
            elif self.model_name in F5TTS_MODELS:
                # Pre-configured model from HuggingFace
                model_config = F5TTS_MODELS[self.model_name]
                
                # Language models need to use base configs but download from custom repos
                if (self.model_name.startswith("F5-") and self.model_name not in ["F5TTS_Base", "F5TTS_v1_Base"]):
                    # Auto-detect model architecture based on name patterns
                    if ("small" in self.model_name.lower() or 
                        "155m" in self.model_name.lower() or
                        self.model_name in ["F5-Hindi-Small"]):  # Known Small models
                        config_name = "F5TTS_Small"  # 18 layers, 768 dim, 12 heads (155M params)
                        print(f"🔍 Detected Small model architecture for '{self.model_name}'")
                    else:
                        config_name = "F5TTS_Base"   # 22 layers, 1024 dim, 16 heads (~1.2GB)
                        print(f"🔍 Using Base model architecture for '{self.model_name}'")
                    repo_id = model_config["repo"]
                    step = model_config["step"]
                    ext = model_config["ext"]
                    
                    # Show download size warning for large models and quality warnings
                    if self.model_name == "F5-JP":
                        print(f"📦 Loading F5-TTS model '{self.model_name}' from {repo_id} using config '{config_name}' (⚠️  Large download: ~5.4GB)")
                    elif self.model_name == "F5-PT-BR":
                        print(f"📦 Loading F5-TTS model '{self.model_name}' from {repo_id} using config '{config_name}' (⚠️  Uses English vocab - may have quality issues)")
                    else:
                        print(f"📦 Loading F5-TTS model '{self.model_name}' from {repo_id} using config '{config_name}'")
                    
                    # Manually construct the model path for custom repo
                    from huggingface_hub import hf_hub_download
                    
                    # Download model and vocab/tokenizer from custom repo
                    # Auto-detect available tokenizer files for each model
                    if self.model_name == "F5-FR":
                        # Use reduced model for F5-FR to save space (1.35GB vs 5.39GB)
                        model_filename = "model_last_reduced.pt"
                        base_path = ""
                        fallback_vocab = "vocab.txt"
                    elif self.model_name == "F5-JP":
                        # Japanese model is in a subfolder with different vocab name
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_{step}.{ext}"
                        base_path = f"{exp_name}/"
                        fallback_vocab = f"{exp_name}/vocab_updated.txt"
                    elif self.model_name == "F5-DE":
                        # German model is in F5TTS_Base subfolder
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_{step}.{ext}"
                        base_path = ""
                        fallback_vocab = "vocab.txt"
                    elif self.model_name == "F5-PT-BR":
                        # Brazilian Portuguese model is in pt-br subfolder
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_last.safetensors"
                        base_path = f"{exp_name}/"
                        fallback_vocab = None  # This model uses original F5-TTS vocab
                    elif self.model_name == "F5-Polish":
                        # Polish multilingual model is in multi3 subfolder
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_{step}.{ext}"
                        base_path = f"{exp_name}/"
                        fallback_vocab = f"{exp_name}/vocab.txt"
                    elif self.model_name == "F5-Hindi-Small":
                        # Hindi Small model from SPRINGLab
                        model_filename = f"model_{step}.{ext}"
                        base_path = ""
                        fallback_vocab = "vocab.txt"
                    else:
                        model_filename = f"model_{step}.{ext}"
                        base_path = ""
                        fallback_vocab = "vocab.txt"

                    # Auto-detect available tokenizer files
                    detected_vocab, detected_tokenizer = detect_tokenizer_files(repo_id, base_path)

                    # Determine which tokenizer files to use (prefer detected over fallback)
                    vocab_filename = detected_vocab or fallback_vocab
                    tokenizer_filename = detected_tokenizer

                    if detected_tokenizer:
                        print(f"🔤 Detected tokenizer.json: {detected_tokenizer}")
                    if detected_vocab:
                        print(f"📝 Detected vocab.txt: {detected_vocab}")
                    elif fallback_vocab:
                        print(f"📝 Using fallback vocab: {fallback_vocab}")
                    
                    # Check if model exists locally first
                    local_model_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", self.model_name, model_filename)
                    
                    if os.path.exists(local_model_path):
                        # print(f"📁 Using local F5-TTS model: {local_model_path}")
                        model_file = local_model_path
                    else:
                        # Check legacy HuggingFace cache location
                        try:
                            hf_cached_file = hf_hub_download(repo_id=repo_id, filename=model_filename, local_files_only=True)
                            # Using cached model silently
                            model_file = hf_cached_file
                        except Exception:
                            # Download to local models directory
                            print(f"📥 Downloading F5-TTS model to local directory: {model_filename}")
                            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                            
                            try:
                                # Use unified downloader
                                from utils.downloads.unified_downloader import unified_downloader
                                
                                files_to_download = [{
                                    'remote': model_filename,
                                    'local': os.path.basename(model_filename)
                                }]
                                
                                downloaded_dir = unified_downloader.download_huggingface_model(
                                    repo_id=repo_id,
                                    model_name=self.model_name,
                                    files=files_to_download,
                                    engine_type="F5-TTS"
                                )
                                
                                if downloaded_dir:
                                    model_file = os.path.join(downloaded_dir, os.path.basename(model_filename))
                                else:
                                    raise Exception("Unified download failed")
                            except Exception as e:
                                # Fallback to HuggingFace cache
                                print(f"⚠️ Failed to download to local directory, using HF cache: {e}")
                                model_file = hf_hub_download(repo_id=repo_id, filename=model_filename)
                    
                    # Handle vocab and tokenizer files - some models don't have their own
                    vocab_file = None
                    tokenizer_file = None

                    if vocab_filename is None and tokenizer_filename is None:
                        # Use original F5-TTS vocab for models that don't have their own
                        # First check if we have F5TTS_Base locally
                        # Try TTS path first, then legacy
                        vocab_search_paths = [
                            os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "F5TTS_Base", "vocab.txt"),
                            os.path.join(folder_paths.models_dir, "F5-TTS", "F5TTS_Base", "vocab.txt"),  # Legacy
                            os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS", "F5TTS_Base", "vocab.txt")  # Legacy
                        ]
                        
                        local_f5tts_base = None
                        for path in vocab_search_paths:
                            if os.path.exists(path):
                                local_f5tts_base = path
                                break
                        if local_f5tts_base:
                            vocab_file = local_f5tts_base
                            print(f"📁 Using local F5TTS_Base vocab: {vocab_file}")
                        else:
                            # Download F5TTS_Base vocab to local directory
                            local_vocab_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "F5TTS_Base", "vocab.txt")
                            
                            if os.path.exists(local_vocab_path):
                                vocab_file = local_vocab_path
                            else:
                                # Check HuggingFace cache first
                                try:
                                    hf_vocab_file = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/vocab.txt", local_files_only=True)
                                    vocab_file = hf_vocab_file
                                    print(f"📁 Using cached F5TTS_Base vocab: {vocab_file}")
                                except Exception:
                                    # Download to local directory
                                    print(f"📥 Downloading F5TTS_Base vocab to local directory")
                                    os.makedirs(os.path.dirname(local_vocab_path), exist_ok=True)
                                    
                                    try:
                                        # Use unified downloader for F5TTS_Base vocab
                                        from utils.downloads.unified_downloader import unified_downloader
                                        
                                        files_to_download = [{
                                            'remote': "F5TTS_Base/vocab.txt",
                                            'local': "vocab.txt"
                                        }]
                                        
                                        downloaded_dir = unified_downloader.download_huggingface_model(
                                            repo_id="SWivid/F5-TTS",
                                            model_name="F5TTS_Base",
                                            files=files_to_download,
                                            engine_type="F5-TTS"
                                        )
                                        
                                        if downloaded_dir:
                                            vocab_file = os.path.join(downloaded_dir, "vocab.txt")
                                        else:
                                            raise Exception("Unified vocab download failed")
                                    except Exception as e:
                                        print(f"⚠️ Failed to download vocab to local directory, using HF cache: {e}")
                                        vocab_file = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/vocab.txt")
                    else:
                        # Model has its own vocab and/or tokenizer files
                        # Download vocab.txt if available
                        if vocab_filename:
                            vocab_file = self._download_tokenizer_file(repo_id, vocab_filename, "vocab")

                        # Download tokenizer.json if available
                        if tokenizer_filename:
                            tokenizer_file = self._download_tokenizer_file(repo_id, tokenizer_filename, "tokenizer")
                    
                    print(f"📁 Downloaded model: {model_file}")
                    if vocab_file:
                        print(f"📁 Downloaded vocab: {vocab_file}")
                    if tokenizer_file:
                        print(f"🔤 Downloaded tokenizer: {tokenizer_file}")

                    # Vocos redirect already setup in _load_f5tts()

                    # Get local Vocos path
                    vocos_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "vocos")
                    vocoder_local_path = vocos_path if os.path.exists(vocos_path) else None

                    # Determine which tokenizer file to use (prefer tokenizer.json over vocab.txt)
                    tokenizer_path = tokenizer_file if tokenizer_file else vocab_file

                    # Load with base config but custom files
                    self.f5tts_model = F5TTS(
                        model=config_name,
                        ckpt_file=model_file,
                        vocab_file=tokenizer_path,  # Pass tokenizer.json or vocab.txt
                        vocoder_local_path=vocoder_local_path,
                        device=self.device
                    )

                    # Store tokenizer information for text processing
                    self.tokenizer_file = tokenizer_file
                    self.vocab_file = vocab_file
                    
                elif self.model_name.startswith("E2-"):
                    # E2 variants use E2 config
                    config_name = "E2TTS_Base"
                    
                    # Check for local E2 model first to avoid Google dependencies
                    model_config = F5TTS_MODELS.get("E2TTS_Base", F5TTS_MODELS["E2TTS_Base"])
                    step = model_config["step"]
                    ext = model_config["ext"]
                    
                    model_filename = f"model_{step}.{ext}"
                    vocab_filename = "vocab.txt"
                    
                    local_model_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "E2TTS_Base", model_filename)
                    local_vocab_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "E2TTS_Base", vocab_filename)
                    
                    if os.path.exists(local_model_path) and os.path.exists(local_vocab_path):
                        print(f"📁 Using local E2-TTS model: {local_model_path}")
                        self.f5tts_model = F5TTS(
                            model=config_name,
                            ckpt_file=local_model_path,
                            vocab_file=local_vocab_path,
                            device=self.device
                        )
                    else:
                        print(f"📦 Loading E2-TTS model '{self.model_name}' from HuggingFace")
                        self.f5tts_model = F5TTS(
                            model=config_name,
                            device=self.device
                        )
                else:
                    # Standard models (F5TTS_Base, F5TTS_v1_Base, E2TTS_Base) - also organize to TTS/ folder
                    print(f"📦 Loading F5-TTS model '{self.model_name}' from HuggingFace")
                    
                    # Get model config for standard models
                    from huggingface_hub import hf_hub_download
                    model_config = F5TTS_MODELS.get(self.model_name, F5TTS_MODELS["F5TTS_Base"])
                    repo_id = model_config["repo"]
                    step = model_config["step"]
                    ext = model_config["ext"]
                    
                    model_filename = f"model_{step}.{ext}"
                    vocab_filename = "vocab.txt"
                    
                    # Check if model exists locally first
                    local_model_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", self.model_name, model_filename)
                    local_vocab_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", self.model_name, vocab_filename)
                    
                    if os.path.exists(local_model_path) and os.path.exists(local_vocab_path):
                        # print(f"📁 Using local F5-TTS model: {local_model_path}")
                        model_file = local_model_path
                        vocab_file = local_vocab_path
                    else:
                        # Download to local models directory for organization
                        print(f"📥 Downloading F5-TTS model to organized directory: {model_filename}")
                        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                        os.makedirs(os.path.dirname(local_vocab_path), exist_ok=True)
                        
                        try:
                            # Use unified downloader to avoid cache duplication
                            from utils.downloads.unified_downloader import unified_downloader
                            
                            # Prepare files to download
                            files_to_download = [
                                {
                                    'remote': f"{self.model_name}/{model_filename}",
                                    'local': model_filename
                                },
                                {
                                    'remote': f"{self.model_name}/{vocab_filename}",
                                    'local': vocab_filename
                                }
                            ]
                            
                            # Download using unified system
                            downloaded_dir = unified_downloader.download_huggingface_model(
                                repo_id=repo_id,
                                model_name=self.model_name,
                                files=files_to_download,
                                engine_type="F5-TTS"
                            )
                            
                            if downloaded_dir:
                                model_file = os.path.join(downloaded_dir, model_filename)
                                vocab_file = os.path.join(downloaded_dir, vocab_filename)
                            else:
                                raise Exception("Unified download failed")
                        except Exception as e:
                            # Fallback to direct F5TTS loading (old behavior)
                            print(f"⚠️ Failed to download to organized directory, using HF cache: {e}")
                            self.f5tts_model = F5TTS(
                                model=self.model_name,
                                device=self.device
                            )
                            model_file = None
                            vocab_file = None
                    
                    # Load using local files if we have them
                    # For E2TTS models, vocab_file may not exist (use empty string like F5TTS API)
                    if model_file:
                        print(f"📁 Downloaded model: {model_file}")
                        print(f"📁 Downloaded vocab: {vocab_file if vocab_file else 'None (using default)'}")
                        
                        # Vocos redirect already setup in _load_f5tts()
                        config_name = self.model_name
                        
                        # Handle missing vocab file by providing explicit path to default vocab
                        if not vocab_file:
                            # Use the bundled vocab file from our F5-TTS installation
                            current_dir = os.path.dirname(__file__)
                            default_vocab = os.path.join(current_dir, "..", "f5_tts", "infer", "examples", "vocab.txt")
                            default_vocab = os.path.normpath(default_vocab)
                            if os.path.exists(default_vocab):
                                vocab_file = default_vocab
                            else:
                                vocab_file = ""  # Fall back to empty string
                        
                        self.f5tts_model = F5TTS(
                            model=config_name,
                            ckpt_file=model_file,
                            vocab_file=vocab_file,
                            device=self.device
                        )
            else:
                # Default fallback using generic utility
                from utils.models.fallback_utils import try_local_first, get_models_dir
                
                # Build search paths for F5-TTS
                search_paths = []
                models_dir = get_models_dir()
                if models_dir:
                    search_paths.append(os.path.join(models_dir, "F5-TTS", "F5TTS_Base"))
                
                def load_local_f5tts(path: str):
                    # Find model and vocab files in local path
                    model_file = None
                    vocab_file = None
                    for file in os.listdir(path):
                        if file.endswith((".safetensors", ".pt")):
                            model_file = os.path.join(path, file)
                        elif file.endswith(".txt") and "vocab" in file.lower():
                            vocab_file = os.path.join(path, file)
                        elif file.endswith(".json") and "tokenizer" in file.lower():
                            vocab_file = os.path.join(path, file)
                    
                    return F5TTS(
                        model="F5TTS_Base",
                        ckpt_file=model_file,
                        vocab_file=vocab_file,
                        device=self.device
                    )
                
                def load_hf_f5tts():
                    # This fallback only runs when local models aren't found
                    print(f"⚠️  Local F5TTS_Base model not found - downloading from HuggingFace")
                    # Get local Vocos path
                    vocos_path = os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "vocos")
                    vocoder_local_path = vocos_path if os.path.exists(vocos_path) else None
                    return F5TTS(model="F5TTS_Base", vocoder_local_path=vocoder_local_path, device=self.device)
                
                self.f5tts_model = try_local_first(
                    search_paths=search_paths,
                    local_loader=load_local_f5tts,
                    fallback_loader=load_hf_f5tts,
                    fallback_name="F5TTS_Base",
                    original_request=self.model_name
                )
                
            print(f"✅ F5-TTS model '{self.model_name}' loaded successfully")
            
        except ImportError as e:
            raise ImportError(f"F5-TTS not available. Please install F5-TTS: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load F5-TTS model '{self.model_name}': {e}")
    
    @classmethod
    def from_local(cls, ckpt_dir: str, device: str, model_name: str = "F5TTS_Base"):
        """Load from local directory following ChatterBox pattern"""
        # Silent loading - only show errors
        return cls(model_name, device, ckpt_dir)
    
    @classmethod  
    def from_pretrained(cls, device: str, model_name: str = "F5TTS_Base"):
        """Load from HuggingFace following ChatterBox pattern"""
        print(f"📦 Loading F5-TTS model '{model_name}' from HuggingFace")
        return cls(model_name, device)
    
    def generate(self, text: str, ref_audio_path: str, ref_text: str, 
                 temperature: float = 0.8, speed: float = 1.0, 
                 target_rms: float = 0.1, cross_fade_duration: float = 0.15,
                 nfe_step: int = 32, cfg_strength: float = 2.0, 
                 auto_phonemization: bool = True, **kwargs) -> torch.Tensor:
        """
        Generate audio with F5-TTS specific parameters
        Following ChatterBox interface pattern
        """
        
        if self.f5tts_model is None:
            raise RuntimeError("F5-TTS model not loaded")
        
        if not ref_text.strip():
            raise ValueError("F5-TTS requires reference text. Please provide ref_text parameter.")
        
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
        
        try:
            # Generate audio using F5-TTS (suppress debug messages, keep progress bars)
            # Set UTF-8 encoding to prevent Windows console encoding issues with international text
            old_env = os.environ.get('PYTHONIOENCODING')
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            try:
                from contextlib import redirect_stdout
                import io
                
                # Suppress stdout debug messages but keep stderr progress bars
                with redirect_stdout(io.StringIO()):
                    wav, sr, _ = self.f5tts_model.infer(
                        ref_file=ref_audio_path,
                        ref_text=ref_text,
                        gen_text=text,
                        target_rms=target_rms,
                        cross_fade_duration=cross_fade_duration,
                        nfe_step=nfe_step,
                        cfg_strength=cfg_strength,
                        speed=speed,
                        remove_silence=False,
                        auto_phonemization=auto_phonemization
                    )
            finally:
                # Restore original encoding
                if old_env is not None:
                    os.environ['PYTHONIOENCODING'] = old_env
                else:
                    os.environ.pop('PYTHONIOENCODING', None)
            
            # Convert to torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            
            # Ensure correct format
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # Add channel dimension
            
            return wav
            
        except Exception as e:
            raise RuntimeError(f"F5-TTS generation failed: {e}")
    
    def prepare_conditionals(self, ref_audio_path: str, ref_text: str):
        """
        Prepare F5-TTS conditionals from reference audio and text
        This is a compatibility method - F5-TTS handles this internally
        """
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
        
        if not ref_text.strip():
            raise ValueError("Reference text cannot be empty for F5-TTS")
        
        # F5-TTS handles preprocessing internally, so we just validate inputs
        return True
    
    def to(self, device):
        """
        Move F5-TTS model to specified device for ComfyUI memory management.
        """
        self.device = device
        
        # Move underlying F5-TTS model components to device
        if hasattr(self, 'f5tts_model') and self.f5tts_model is not None:
            # F5-TTS model has multiple components that need to be moved
            if hasattr(self.f5tts_model, 'ema_model') and self.f5tts_model.ema_model is not None:
                self.f5tts_model.ema_model.to(device)
            if hasattr(self.f5tts_model, 'vocoder') and self.f5tts_model.vocoder is not None:
                self.f5tts_model.vocoder.to(device)
        
        return self
    
    def edit_speech(self, audio_tensor: torch.Tensor, sample_rate: int,
                   original_text: str, target_text: str,
                   edit_regions: list, fix_durations: list = None,
                   temperature: float = 0.8, speed: float = 1.0,
                   target_rms: float = 0.1, nfe_step: int = 32,
                   cfg_strength: float = 2.0, sway_sampling_coef: float = -1.0,
                   ode_method: str = "euler", **kwargs) -> torch.Tensor:
        """
        Edit speech using F5-TTS speech editing functionality
        
        Args:
            audio_tensor: Original audio tensor
            sample_rate: Sample rate of the audio
            original_text: Original text that matches the audio
            target_text: Target text with desired changes
            edit_regions: List of [start, end] time regions to edit (in seconds)
            fix_durations: Optional list of fixed durations for each edit region
            temperature: Sampling temperature
            speed: Speech speed multiplier
            target_rms: Target RMS level
            nfe_step: Number of function evaluations
            cfg_strength: CFG strength
            sway_sampling_coef: Sway sampling coefficient
            ode_method: ODE integration method
            
        Returns:
            Edited audio tensor
        """
        if self.f5tts_model is None:
            raise RuntimeError("F5-TTS model not loaded")
        
        if not original_text.strip():
            raise ValueError("Original text cannot be empty")
        
        if not target_text.strip():
            raise ValueError("Target text cannot be empty")
        
        if not edit_regions:
            raise ValueError("Edit regions cannot be empty")
        
        try:
            # This is a placeholder for the actual speech editing implementation
            # The actual implementation would need to use F5-TTS's speech editing capabilities
            # For now, we'll use the regular generation as a fallback
            print(f"⚠️ Speech editing not yet implemented in F5-TTS API wrapper")
            print(f"Falling back to regular generation with target text: {target_text}")
            
            # Save audio to temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Ensure audio is in correct format
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Save the audio
            import torchaudio
            torchaudio.save(temp_path, audio_tensor, sample_rate)
            
            # Use the regular generation as a fallback
            wav = self.generate(
                text=target_text,
                ref_audio_path=temp_path,
                ref_text=original_text,
                temperature=temperature,
                speed=speed,
                target_rms=target_rms,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                **kwargs
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return wav
            
        except Exception as e:
            raise RuntimeError(f"F5-TTS speech editing failed: {e}")


def find_f5tts_models():
    """
    Find F5-TTS model files in order of priority.
    Returns list of tuples containing (source_type, path)
    """
    model_paths = []
    
    # 1. Check ComfyUI models folder - F5-TTS directory
    comfyui_f5tts_path = os.path.join(folder_paths.models_dir, "F5-TTS")
    if os.path.exists(comfyui_f5tts_path):
        for item in os.listdir(comfyui_f5tts_path):
            item_path = os.path.join(comfyui_f5tts_path, item)
            if os.path.isdir(item_path):
                # Check if it contains model files
                has_model = False
                for ext in [".safetensors", ".pt"]:
                    if any(f.endswith(ext) for f in os.listdir(item_path)):
                        has_model = True
                        break
                if has_model:
                    model_paths.append(("comfyui", item_path))
    
    # 2. HuggingFace download as fallback
    model_paths.append(("huggingface", None))
    
    return model_paths