from pathlib import Path
import os

import librosa
import torch
import warnings
from huggingface_hub import hf_hub_download

# Import folder_paths for model directory detection
try:
    import folder_paths
except ImportError:
    folder_paths = None

# Import perth with warnings disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import perth

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.watermarker = perth.PerthImplicitWatermarker()
        self.enable_watermarking = False  # Disabled by default for maximum compatibility
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        print(f"📦 Loading local ChatterBox VC models from: {ckpt_dir}")
        ckpt_dir = Path(ckpt_dir)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            ref_dict = None
            if (builtin_voice := ckpt_dir / "conds.pt").exists():
                states = torch.load(builtin_voice)
                ref_dict = states['gen']

            s3gen = S3Gen()
            s3gen.load_state_dict(
                torch.load(ckpt_dir / "s3gen.pt")
            )
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
        
        # Download VC models to local directory
        local_paths = []
        for fpath in ["s3gen.pt", "conds.pt"]:
            # Define local path in TTS organization
            local_model_path = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", language, fpath)
            
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
            except Exception:
                pass
            
            # Download to local directory
            print(f"📥 Downloading Chatterbox VC model to local directory: {fpath}")
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            
            try:
                temp_file = hf_hub_download(repo_id=repo_id, filename=fpath)
                import shutil
                shutil.copy2(temp_file, local_model_path)
                local_paths.append(local_model_path)
                print(f"✅ Downloaded Chatterbox VC model to: {local_model_path}")
            except Exception as e:
                # Fallback to HuggingFace cache
                print(f"⚠️ Failed to download to local directory, using HF cache: {e}")
                local_path = hf_hub_download(repo_id=repo_id, filename=fpath)
                local_paths.append(local_path)

        # Use the directory of the first downloaded file
        model_dir = Path(local_paths[0]).parent
        return cls.from_local(model_dir, device)

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

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
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            if self.enable_watermarking:
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)
