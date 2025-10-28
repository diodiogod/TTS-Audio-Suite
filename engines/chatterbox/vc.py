from pathlib import Path
import os

import torch
import warnings
from huggingface_hub import hf_hub_download

# Use librosa fallback for Python 3.13 compatibility
from utils.audio.librosa_fallback import safe_load

# Import folder_paths for model directory detection
try:
    import folder_paths
except ImportError:
    folder_paths = None

# Import perth with warnings disabled and graceful fallback
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import perth
    PERTH_AVAILABLE = True
except ImportError:
    perth = None
    PERTH_AVAILABLE = False

# Import safetensors for multilanguage model support
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        # Initialize watermarker silently (but disabled by default)
        if PERTH_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.watermarker = perth.PerthImplicitWatermarker()
        else:
            self.watermarker = None
        self.enable_watermarking = False  # Disabled by default for maximum compatibility
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.
        """
        self.device = device

        # Move s3gen model
        if hasattr(self.s3gen, 'to'):
            self.s3gen = self.s3gen.to(device)

        # Move reference dict tensors
        if self.ref_dict is not None:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in self.ref_dict.items()
            }

        return self

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        print(f"📦 Loading local ChatterBox VC models from: {ckpt_dir}")
        ckpt_dir = Path(ckpt_dir)
        
        # Auto-detect model format (same as TTS implementation)
        def load_model_file(base_name: str):
            """Load model file with auto-detection of format (.safetensors preferred over .pt)"""
            safetensors_path = ckpt_dir / f"{base_name}.safetensors"
            pt_path = ckpt_dir / f"{base_name}.pt"
            
            if safetensors_path.exists():
                return load_file(safetensors_path, device=device)
            elif pt_path.exists():
                return torch.load(pt_path, map_location=device)
            else:
                raise FileNotFoundError(f"Neither {base_name}.safetensors nor {base_name}.pt found in {ckpt_dir}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            ref_dict = None
            if (builtin_voice := ckpt_dir / "conds.pt").exists():
                states = torch.load(builtin_voice)
                ref_dict = states['gen']

            s3gen = S3Gen()
            # Load s3gen with auto-format detection
            s3gen_state = load_model_file("s3gen")
            # Apply JaneDoe84's critical fix: strict=False to handle missing keys (like Norwegian model)
            s3gen.load_state_dict(s3gen_state, strict=False)
            s3gen.to(device).eval()

            instance = cls(s3gen, device, ref_dict=ref_dict)
            print("✅ Successfully loaded all local ChatterBox VC models")
            return instance

    @classmethod
    def from_pretrained(cls, device, language="English") -> 'ChatterboxVC':
        """
        Load ChatterBox VC model from HuggingFace Hub with language support.
        
        Args:
            device: Target device
            language: Language model to load (English, German, Norwegian, etc.)
            
        Returns:
            ChatterboxVC model instance
        """
        # Handle local models first - redirect to from_local
        if language.startswith("local:"):
            from .language_models import find_local_model_path
            local_path = find_local_model_path(language)
            if local_path and os.path.exists(local_path):
                print(f"📁 Redirecting to local model at: {local_path}")
                return cls.from_local(local_path, device)
            else:
                print(f"⚠️ Local model path not found for {language}, falling back to English")
                language = "English"
        
        # Import language model support
        try:
            from .language_models import get_model_config
            
            # Get model configuration for the specified language  
            model_config = get_model_config(language)
            if not model_config:
                print(f"⚠️ Language '{language}' not found, falling back to English")
                model_config = get_model_config("English")
                if not model_config:
                    # Final fallback to original repo
                    repo_id = REPO_ID
                else:
                    repo_id = model_config["repo"]
            else:
                repo_id = model_config["repo"]
                
        except ImportError:
            # Fallback if language_models not available
            print(f"⚠️ Language models not available, using English model")
            repo_id = REPO_ID
        
        print(f"📦 Loading ChatterBox VC model for {language} from {repo_id}")
        
        # Sanitize language name for path construction (strip "local:" prefix)
        clean_language = language[6:] if language.startswith("local:") else language
        
        # Download VC models to local directory (use dynamic requirements like TTS)
        try:
            from .language_models import get_model_requirements, get_model_config
            # Get VC-specific files (filter out TTS-only files) 
            all_files = get_model_requirements(language)
            vc_files = [f for f in all_files if any(f.startswith(component) for component in ["s3gen", "conds"])]
            
            # Separate required vs optional files (like TTS does)
            required_files = [f for f in vc_files if f.startswith("s3gen")]  # s3gen is required
            optional_files = [f for f in vc_files if f.startswith("conds")]   # conds is optional
            
            # Get subdirectory info like TTS does
            model_config = get_model_config(language)
            subdirectory = model_config.get("subdirectory") if model_config else None
            
        except ImportError:
            # Fallback for older systems
            required_files = ["s3gen.pt"]
            optional_files = ["conds.pt"]
            subdirectory = None
        
        local_paths = []
        all_vc_files = required_files + optional_files
        
        for fpath in all_vc_files:
            # Define local path in TTS organization (use clean language name)
            local_model_path = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", clean_language, fpath)
            
            if os.path.exists(local_model_path):
                print(f"📁 Using local Chatterbox VC model: {local_model_path}")
                local_paths.append(local_model_path)
                continue
            
            # Check HuggingFace cache first
            try:
                hf_cached_file = hf_hub_download(repo_id=repo_id, filename=fpath, local_files_only=True)
                print(f"📁 Using cached Chatterbox VC model: {hf_cached_file}")
                local_paths.append(hf_cached_file)
                continue
            except Exception as cache_error:
                print(f"📋 Cache check for {fpath}: {str(cache_error)[:100]}... - will download")
                pass
            
            # Download to local directory
            print(f"📥 Downloading Chatterbox VC model to local directory: {fpath}")
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            
            try:
                # Use unified downloader with subdirectory support
                from utils.downloads.unified_downloader import unified_downloader
                
                # Build correct remote path with subdirectory
                remote_path = fpath
                if subdirectory:
                    remote_path = f"{subdirectory}/{fpath}"
                
                files_to_download = [{
                    'remote': remote_path,
                    'local': os.path.basename(fpath)  # Save directly to language folder
                }]
                
                downloaded_dir = unified_downloader.download_huggingface_model(
                    repo_id=repo_id,
                    model_name=clean_language,
                    files=files_to_download,
                    engine_type="chatterbox"
                )
                
                if downloaded_dir:
                    local_model_path = os.path.join(downloaded_dir, os.path.basename(fpath))
                    local_paths.append(local_model_path)
                    print(f"✅ Downloaded Chatterbox VC model to: {local_model_path}")
                else:
                    # Check if this is an optional file before raising error
                    is_optional = fpath in optional_files
                    if is_optional:
                        print(f"ℹ️ Optional file {fpath} not available in {language} model - continuing without it")
                        continue
                    else:
                        raise Exception("Unified download failed")
            except Exception as e:
                # Check for 401 authorization errors - these should be propagated
                error_str = str(e)
                if "401" in error_str or "Unauthorized" in error_str:
                    raise RuntimeError(f"401 Unauthorized: {language} model requires authentication - {e}")
                
                # Fallback to HuggingFace cache for other errors
                print(f"⚠️ Failed to download to local directory, using HF cache: {e}")
                try:
                    local_path = hf_hub_download(repo_id=repo_id, filename=fpath)
                    local_paths.append(local_path)
                except Exception as hf_error:
                    # Check HF cache error for 401 as well
                    hf_error_str = str(hf_error)
                    if "401" in hf_error_str or "Unauthorized" in hf_error_str:
                        raise RuntimeError(f"401 Unauthorized: {language} model requires authentication - {hf_error}")
                    else:
                        # Handle missing files based on whether they're required or optional
                        is_optional = fpath in optional_files
                        if is_optional and ("404" in hf_error_str or "Not Found" in hf_error_str):
                            print(f"ℹ️ Optional file {fpath} not available in {language} model - continuing without it")
                            continue  # Skip optional files that are missing
                        else:
                            raise hf_error  # Re-raise errors for required files

        # Log final source summary
        sources = []
        for path in local_paths:
            if "models/TTS/chatterbox" in str(path):
                sources.append("local")
            elif "cache" in str(path).lower():
                sources.append("cache") 
            else:
                sources.append("downloaded")
        print(f"📦 Loading ChatterBox VC using: {', '.join(set(sources))} sources")
        
        # Use the directory of the first downloaded file
        model_dir = Path(local_paths[0]).parent
        return cls.from_local(model_dir, device)

    def set_target_voice(self, wav_fpath):
        # Load reference wav using fallback for Python 3.13 compatibility
        s3gen_ref_wav, sample_rate = safe_load(wav_fpath, sr=S3GEN_SR, mono=True)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            # Load audio using fallback for Python 3.13 compatibility
            audio_16_wav, _ = safe_load(audio, sr=S3_SR, mono=True)
            # Convert to the expected format
            audio_16 = torch.from_numpy(audio_16_wav).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            if self.enable_watermarking and self.watermarker is not None:
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)
