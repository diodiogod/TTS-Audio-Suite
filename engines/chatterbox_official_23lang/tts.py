from dataclasses import dataclass
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import warnings

# Use librosa fallback for Python 3.13 compatibility
from utils.audio.librosa_fallback import safe_load, safe_resample
# Import safetensors for multilanguage model support
from safetensors.torch import load_file

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

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer, MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# Import language model registry for Official 23-Lang
try:
    from .language_models import (
        get_model_config, OFFICIAL_23LANG_MODELS, get_model_requirements, 
        validate_model_completeness, get_supported_languages, SUPPORTED_LANGUAGES
    )
except ImportError:
    # Fallback if language_models not available
    OFFICIAL_23LANG_MODELS = {"ChatterBox Official 23-Lang": {"repo": "ResembleAI/chatterbox", "format": "mixed"}}
    SUPPORTED_LANGUAGES = {"en": "English"}
    def get_model_config(model_name):
        return OFFICIAL_23LANG_MODELS.get(model_name, OFFICIAL_23LANG_MODELS["ChatterBox Official 23-Lang"])
    def get_model_requirements(model_name, model_version="v1"):
        return ["t3_23lang.safetensors", "s3gen.pt", "ve.pt", "mtl_tokenizer.json", "conds.pt"]
    def validate_model_completeness(model_path, model_name, model_version="v1"):
        return True, []
    def get_supported_languages():
        return list(SUPPORTED_LANGUAGES.keys())

# Import modular processors
from .overlapping_processor import OverlappingBatchProcessor, BatchingStrategy

REPO_ID = "ResembleAI/chatterbox"  # Default for backward compatibility


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        # CRITICAL: Also move s3gen model to device (fixes speaker_encoder device mismatch)
        if hasattr(self, 's3gen') and self.s3gen is not None:
            self.s3gen = self.s3gen.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxOfficial23LangTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer,  # Union[EnTokenizer, MTLTokenizer]
        device: str,
        conds: Conditionals = None,
        model_version: str = "v1",
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.model_version = model_version  # v1 or v2
        # Watermarking disabled by default for maximum compatibility
        # Set to True to enable watermarking (requires perth library)
        self.enable_watermarking = False
        self.watermarker = None
        self._watermarker_init_attempted = False

        # Initialize modular processors
        self.overlapping_processor = OverlappingBatchProcessor(self)
    
    def _init_watermarker_if_needed(self):
        """Initialize watermarker on first use if enabled"""
        if self.enable_watermarking and not self._watermarker_init_attempted:
            self._watermarker_init_attempted = True
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.watermarker = perth.PerthImplicitWatermarker()
                    if self.watermarker is None:
                        raise ValueError("PerthImplicitWatermarker returned None")
            except Exception as e:
                print(f"❌ Failed to initialize watermarker: {e}")
                self.watermarker = None
                self.enable_watermarking = False

    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.
        """
        self.device = device

        # Move all PyTorch model components
        if hasattr(self.t3, 'to'):
            self.t3 = self.t3.to(device)
        if hasattr(self.s3gen, 'to'):
            self.s3gen = self.s3gen.to(device)
        if hasattr(self.ve, 'to'):
            self.ve = self.ve.to(device)
        if hasattr(self.tokenizer, 'to'):
            self.tokenizer = self.tokenizer.to(device)
        if self.conds is not None and hasattr(self.conds, 'to'):
            self.conds = self.conds.to(device)

        return self

    @classmethod
    def from_local(cls, ckpt_dir, device, model_name=None, model_version="v2") -> 'ChatterboxOfficial23LangTTS':
        """
        Load ChatterBox Official 23-Lang multilingual model from local directory.
        Expected files:
        - t3_23lang.safetensors (multilingual T3 model v1) OR t3_mtl23ls_v2.safetensors (v2)
        - s3gen.pt (S3Gen model)
        - ve.pt (Voice encoder)
        - mtl_tokenizer.json (multilingual tokenizer)
        - grapheme_mtl_merged_expanded_v1.json (v2 enhanced tokenization, optional)
        - conds.pt (optional conditioning)
        """
        ckpt_dir = Path(ckpt_dir)
        print(f"📦 Loading ChatterBox Official 23-Lang model from: {ckpt_dir}")
        
        # Resolve "auto" device to actual device
        if device == "auto":
            if torch.cuda.is_available():
                actual_device = "cuda"
            else:
                actual_device = "cpu"
        else:
            actual_device = device
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        # (Following official ResembleAI implementation)
        if actual_device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        # Check model completeness with version
        if model_name:
            is_complete, missing_files = validate_model_completeness(str(ckpt_dir), model_name, model_version)
            if not is_complete:
                print(f"⚠️ Missing {model_version} files: {missing_files}")
                # Continue loading anyway - some files like conds.pt are optional
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load Voice Encoder (try safetensors first, fallback to pt)
            print("📦 Loading Voice Encoder...")
            ve_safetensors_path = ckpt_dir / "ve.safetensors"
            ve_pt_path = ckpt_dir / "ve.pt"
            
            ve = VoiceEncoder()
            # Smart loading: prefer official .pt, but support user's safetensors
            has_safetensors = ve_safetensors_path.exists()
            has_pt = ve_pt_path.exists()
            
            if has_pt:
                # Use official .pt format (guaranteed compatibility)
                ve_state = torch.load(ve_pt_path, map_location=map_location, weights_only=True)
                ve.load_state_dict(ve_state)
            elif has_safetensors:
                # User manually downloaded safetensors - use with compatibility warning
                print("⚠️ Using ve.safetensors with strict=False for compatibility. Official implementation uses .pt files")
                ve_state = load_file(ve_safetensors_path, device=actual_device)
                ve.load_state_dict(ve_state, strict=False)
            else:
                raise FileNotFoundError(f"Voice encoder not found: tried {ve_safetensors_path} and {ve_pt_path}")
            
            ve.to(actual_device).eval()
            
            # Load T3 multilingual model based on version
            print(f"📦 Loading T3 23-Lang {model_version} model...")
            if model_version == "v2":
                t3_path = ckpt_dir / "t3_mtl23ls_v2.safetensors"
            else:
                t3_path = ckpt_dir / "t3_23lang.safetensors"

            if not t3_path.exists():
                raise FileNotFoundError(f"T3 multilingual {model_version} model not found: {t3_path}")
            
            # Load T3 with version-specific multilingual config
            from .models.t3.t3 import T3Config
            config = T3Config.multilingual(version=model_version)
            t3 = T3(config)
            
            t3_state = load_file(t3_path, device=actual_device)
            if "model" in t3_state.keys():
                t3_state = t3_state["model"][0]
            
            t3.load_state_dict(t3_state)
            t3.to(actual_device).eval()
            
            # Load S3Gen (try safetensors first, fallback to pt)
            print("📦 Loading S3Gen model...")
            s3gen_safetensors_path = ckpt_dir / "s3gen.safetensors"
            s3gen_pt_path = ckpt_dir / "s3gen.pt"
            
            s3gen = S3Gen()
            # Smart loading: prefer official .pt, but support user's safetensors
            has_safetensors = s3gen_safetensors_path.exists()
            has_pt = s3gen_pt_path.exists()
            
            if has_pt:
                # Use official .pt format (guaranteed compatibility)
                s3gen_state = torch.load(s3gen_pt_path, map_location=map_location, weights_only=True)
                s3gen.load_state_dict(s3gen_state)
            elif has_safetensors:
                # User manually downloaded safetensors - use with compatibility warning
                print("⚠️ Using s3gen.safetensors with strict=False for compatibility. Official implementation uses .pt files")
                s3gen_state = load_file(s3gen_safetensors_path, device=actual_device)
                s3gen.load_state_dict(s3gen_state, strict=False)
            else:
                raise FileNotFoundError(f"S3Gen model not found: tried {s3gen_safetensors_path} and {s3gen_pt_path}")
            
            s3gen.to(actual_device).eval()
            
            # Load version-specific multilingual tokenizer
            print("📦 Loading multilingual tokenizer...")
            if model_version == "v2":
                tokenizer_path = ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"
                if not tokenizer_path.exists():
                    # Fallback to v1 tokenizer if v2 tokenizer not found
                    print("⚠️ v2 tokenizer not found, falling back to v1 tokenizer")
                    tokenizer_path = ckpt_dir / "mtl_tokenizer.json"
            else:
                tokenizer_path = ckpt_dir / "mtl_tokenizer.json"

            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Multilingual tokenizer not found: {tokenizer_path}")

            tokenizer = MTLTokenizer(str(tokenizer_path))
            
            # Load conditioning (optional)
            conds = None
            conds_path = ckpt_dir / "conds.pt"
            if conds_path.exists():
                print("📦 Loading conditioning...")
                conds = Conditionals.load(conds_path).to(actual_device)
            else:
                print("⚠️ No conditioning found - using None")
            
            # Create instance
            instance = cls(t3, s3gen, ve, tokenizer, actual_device, conds=conds, model_version=model_version)
            
            # Show model path info
            print(f"📦 Model loaded from: {ckpt_dir}")
            
            return instance

    @classmethod
    def _load_unified_model(cls, ckpt_dir, device, language) -> 'ChatterboxTTS':
        """
        Load unified model (like Italian) that contains all components in a single checkpoint file.
        Based on the deployment script from niobures/Chatterbox-TTS/it,en/deploy_italian_tts.py
        """
        ckpt_dir = Path(ckpt_dir)
        print(f"🇮🇹 Loading unified {language} ChatterBox model from: {ckpt_dir}")
        
        # Resolve "auto" device to actual device
        if device == "auto":
            if torch.cuda.is_available():
                actual_device = "cuda"
            else:
                actual_device = "cpu"
        else:
            actual_device = device
        
        # Load the unified model file
        unified_model_path = ckpt_dir / "chatterbox_italian_final.pt"
        if not unified_model_path.exists():
            raise FileNotFoundError(f"Unified model file not found: {unified_model_path}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            print(f"📦 Loading checkpoint: {unified_model_path.name}")
            checkpoint = torch.load(unified_model_path, map_location=actual_device)
            
            # Extract model configuration
            model_config = checkpoint.get('model_config', {})
            vocab_size = model_config.get('vocab_size', 1500)
            frozen_embeddings = model_config.get('frozen_embeddings', 704)
            trainable_tokens = model_config.get('trainable_tokens', 796)
            
            print(f"🎯 Vocab size: {vocab_size}")
            print(f"🔒 Frozen tokens: {frozen_embeddings}")
            print(f"🔥 Trainable tokens: {trainable_tokens}")
            
            # Load base English ChatterBox components first
            print(f"📦 Loading base English model components...")
            
            # Initialize components with standard ChatterBox architecture
            ve = VoiceEncoder()
            ve.load_state_dict(checkpoint['ve_state_dict'], strict=False)
            ve.to(actual_device).eval()
            
            # Load T3 config with Italian vocabulary size
            from .models.t3.t3 import T3Config
            config = T3Config()
            # Override text token dictionary size for Italian extended vocab
            config.text_tokens_dict_size = vocab_size  # Use the 1500 tokens from checkpoint
            t3 = T3(config)
            t3.load_state_dict(checkpoint['t3_state_dict'], strict=False)
            t3.tfmr.output_attentions = False
            t3.to(actual_device).eval()
            
            # Load S3Gen
            s3gen = S3Gen()
            s3gen.load_state_dict(checkpoint['s3gen_state_dict'], strict=False)
            s3gen.to(actual_device).eval()
            
            # Load tokenizer for Italian model (fallback to English tokenizer)
            # The Italian model uses language prefixes like [it] for Italian text
            from .models.tokenizers import EnTokenizer
            from .language_models import find_local_model_path
            
            # Try to use existing tokenizer if available in Italian model
            tokenizer_path = ckpt_dir / "tokenizer.json"
            if tokenizer_path.exists():
                tokenizer = EnTokenizer(str(tokenizer_path))
                print(f"🔤 Using Italian model tokenizer: tokenizer.json")
            else:
                # Fallback: use English model tokenizer (Italian extends English vocab)
                english_model_path = find_local_model_path("English")
                if english_model_path:
                    english_tokenizer_path = Path(english_model_path) / "tokenizer.json"
                    if english_tokenizer_path.exists():
                        tokenizer = EnTokenizer(str(english_tokenizer_path))
                        print(f"🔤 Using English tokenizer fallback: {english_tokenizer_path}")
                    else:
                        raise FileNotFoundError(f"No tokenizer found for Italian model and no English fallback available")
                else:
                    raise FileNotFoundError(f"No tokenizer available for Italian model and no English model found for fallback")
            
            # Check for conditional voices
            conds = None
            if (builtin_voice := ckpt_dir / "conds.pt").exists():
                conds = Conditionals.load(builtin_voice).to(actual_device)
            
            instance = cls(t3, s3gen, ve, tokenizer, actual_device, conds=conds, model_version=model_version)
            print(f"✅ Italian TTS model loaded successfully!")
            print(f"💡 Use '[it]' prefix for Italian text, no prefix for English")
            print(f"📦 Model loaded from: {ckpt_dir}")
            return instance

    @classmethod
    def from_pretrained(cls, device, model_name="ChatterBox Official 23-Lang", model_version="v2") -> 'ChatterboxOfficial23LangTTS':
        """
        Download and load ChatterBox Official 23-Lang model from HuggingFace Hub.

        Args:
            device: Device to load model on
            model_name: Model to load (defaults to "ChatterBox Official 23-Lang")
            model_version: Model version - "v1" or "v2" (defaults to "v2")
        """
        # Get model configuration
        model_config = get_model_config(model_name)
        if not model_config:
            print(f"⚠️ Model '{model_name}' not found, using default")
            model_name = "ChatterBox Official 23-Lang"
            model_config = get_model_config(model_name)
            # If falling back to English, try local first
            if language != "English":
                from utils.models.fallback_utils import try_local_first, get_models_dir
                
                # Build search paths for ChatterBox
                search_paths = []
                models_dir = get_models_dir()
                if models_dir:
                    search_paths.append(os.path.join(models_dir, "chatterbox"))
                
                # Add common fallback paths
                search_paths.extend([
                    os.path.join(os.getcwd(), "models", "chatterbox"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "models", "chatterbox")
                ])
                
                try:
                    return try_local_first(
                        search_paths=search_paths,
                        local_loader=lambda path: cls.from_local(path, device, language),
                        fallback_loader=lambda: cls.from_pretrained(device, language="English"),
                        fallback_name="English",
                        original_request=language
                    )
                except Exception as e:
                    print(f"⚠️ Fallback failed: {e}, proceeding with direct HuggingFace download")
            
            model_config = get_model_config("English")
        
        repo_id = model_config.get("repo", REPO_ID)
        model_format = model_config.get("format", "pt")
        
        # Silent loading unless errors occur
        
        # Check if local model exists first
        model_dir = os.path.join(folder_paths.models_dir, "TTS", "chatterbox_official_23lang", model_name)
        if os.path.exists(model_dir):
            # Validate model completeness with version
            is_complete, missing_files = validate_model_completeness(model_dir, model_name, model_version)
            if is_complete:
                print(f"📁 Using existing local {model_version} model: {model_dir}")
                return cls.from_local(model_dir, device, model_name, model_version)
            else:
                print(f"⚠️ Local {model_version} model incomplete, missing: {missing_files}")
                print(f"📥 Downloading missing {model_version} files...")

        # Use new unified ChatterBox downloader
        from utils.downloads.unified_downloader import unified_downloader

        # Get files to download based on model version
        files_to_download = get_model_requirements(model_name, model_version)
        
        # Handle mixed format models (try safetensors first, fallback to pt)
        # Exception: conds.pt doesn't have a safetensors version, keep as .pt
        if model_format == "mixed":
            # Try safetensors first, but exclude files that only exist as .pt
            files_to_download = [
                f.replace('.pt', '.safetensors') if (f.endswith('.pt') and f != 'conds.pt') else f
                for f in files_to_download
            ]
        elif model_format == "pt":
            files_to_download = [f.replace('.safetensors', '.pt') if f.endswith('.safetensors') else f for f in files_to_download]
        
        # Normalize model name for download path (strip "ChatterBox " prefix if present)
        # This ensures consistent paths: "ChatterBox Official 23-Lang" → "Official 23-Lang"
        download_model_name = model_name.replace("ChatterBox ", "") if model_name.startswith("ChatterBox ") else model_name

        # Download model using new method with subdirectory support
        subdirectory = model_config.get("subdirectory")
        downloaded_dir = unified_downloader.download_chatterbox_model(
            repo_id=repo_id,
            model_name=download_model_name,
            subdirectory=subdirectory,
            files=files_to_download
        )

        if downloaded_dir:
            print(f"✅ Downloaded ChatterBox {model_version} model files")
            return cls.from_local(downloaded_dir, device, model_name, model_version)
        else:
            # Download failed
            raise Exception(f"Failed to download ChatterBox Official 23-Lang {model_version} model files")

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        # Load reference wav using fallback for Python 3.13 compatibility
        s3gen_ref_wav, sample_rate = safe_load(wav_fpath, sr=S3GEN_SR, mono=True)
        
        # Resample to 16k for S3 tokenizer using fallback
        ref_16k_wav = safe_resample(s3gen_ref_wav, S3GEN_SR, S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        """Generate audio for a single text input with specified language."""
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # CRITICAL FIX: Ensure conditioning data is on the correct device (fixes device mismatch when reusing cached models)
        # When models are moved from CPU back to GPU, the embedding tensors are moved but the conditioning data may still be on CPU
        if self.conds is not None and hasattr(self.conds, 'to'):
            self.conds = self.conds.to(device=self.device)

        # Update exaggeration if needed
        if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text with language-specific processing
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            if self.enable_watermarking:
                self._init_watermarker_if_needed()
                if self.watermarker is not None:
                    watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                    return torch.from_numpy(watermarked_wav).unsqueeze(0)
            
            return torch.from_numpy(wav).unsqueeze(0)
    
    def generate_batch(
        self,
        texts,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
        batch_size=4,
        max_workers=None,  # Allow configurable max_workers
        enable_adaptive_batching=False,  # NEW: Enable adaptive processing
    ):
        """
        Generate audio for multiple text inputs using TRUE batched processing.

        FIXED: Now processes multiple texts simultaneously using batch inference,
        not sequential loops like before.

        Args:
            texts: List of text strings to generate audio for
            audio_prompt_path: Path to reference audio
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            batch_size: Number of texts to process in parallel

        Returns:
            List of audio tensors
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # CRITICAL FIX: Ensure conditioning data is on the correct device (fixes device mismatch when reusing cached models)
        # When models are moved from CPU back to GPU, the embedding tensors are moved but the conditioning data may still be on CPU
        if self.conds is not None and hasattr(self.conds, 'to'):
            self.conds = self.conds.to(device=self.device)

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        
        print(f"🚀 ChatterBox TRUE BATCH processing: {len(texts)} texts")
        
        effective_max_workers = max_workers or batch_size
        
        # NEW: CHOOSE PROCESSING STRATEGY
        if effective_max_workers <= 1:
            # Force sequential processing for 0-1 workers (no threading overhead)
            print(f"→ SEQUENTIAL PROCESSING: {len(texts)} texts (workers={effective_max_workers})")
            results = []
            for i, text in enumerate(texts):
                print(f"  🎤 Sequential {i+1}/{len(texts)}: {text[:40]}...")
                try:
                    individual_audio = self.generate(
                        text=text,
                        audio_prompt_path=None,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )
                    results.append(individual_audio)
                    print(f"  ✅ Sequential {i+1}: Completed")
                except Exception as individual_error:
                    print(f"  ❌ Sequential {i+1}: Failed - {individual_error}")
                    results.append(torch.zeros(1, 1000))
        elif enable_adaptive_batching:
            # Use NEW continuous processor for true non-stop parallelization
            print(f"🌊 Using CONTINUOUS processing (workers never idle)")
            results = self.continuous_processor.process_texts_continuously(
                texts, temperature, cfg_weight, exaggeration, effective_max_workers
            )
        elif len(texts) <= effective_max_workers:
            # All texts fit in one batch - process simultaneously  
            print(f"🚀 SINGLE BATCH: Processing all {len(texts)} texts simultaneously")
            try:
                results = self._batch_inference_simultaneous(
                    texts, temperature, cfg_weight, effective_max_workers
                )
                print(f"✅ All texts completed successfully")
            except Exception as e:
                print(f"⚠️ Batch inference failed: {e}")
                print(f"🔄 Falling back to individual processing")
                results = []
                for text in texts:
                    try:
                        individual_audio = self.generate(
                            text=text,
                            audio_prompt_path=None,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                        )
                        results.append(individual_audio)
                    except Exception as individual_error:
                        print(f"❌ Individual generation failed for text: {text[:50]}...")
                        results.append(torch.zeros(1, 1000))
        else:
            # Use modular overlapping processor for fixed throughput
            strategy = BatchingStrategy.choose_strategy(len(texts), effective_max_workers)
            print(f"🔥 STRATEGY: {BatchingStrategy.get_strategy_description(strategy)}")
            
            results = self.overlapping_processor.process_texts_with_overlap(
                texts, temperature, cfg_weight, exaggeration, effective_max_workers
            )
        
        return results
    
    def _batch_inference_simultaneous(self, texts, temperature=0.8, cfg_weight=0.5, max_workers=4):
        """
        TRUE PARALLEL PROCESSING: Like Chatterbox-TTS-Extended does it!
        
        Uses ThreadPoolExecutor to process multiple texts in parallel threads.
        Each thread uses the standard generate() method with full CFG support.
        This is the CORRECT way to do batch processing with ChatterBox.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"🔥 TRUE PARALLEL PROCESSING: {len(texts)} texts in parallel threads")
        
        def _process_single_text(text_index, text):
            """Process a single text in a separate thread."""
            try:
                print(f"  🧵 Thread {text_index+1}: Starting generation for: {text[:30]}...")
                
                # Use the existing, working generate method with FULL CFG support
                audio = self.generate(
                    text=text,
                    audio_prompt_path=None,  # Use pre-loaded conditioning
                    exaggeration=self.conds.t3.emotion_adv[0, 0, 0].item(),
                    cfg_weight=cfg_weight,  # KEEP CFG for quality!
                    temperature=temperature,
                )
                
                print(f"  ✅ Thread {text_index+1}: Completed successfully")
                return text_index, audio
                
            except Exception as e:
                print(f"  ❌ Thread {text_index+1}: Failed with error: {e}")
                return text_index, torch.zeros(1, 1000)  # Return empty audio on failure
        
        # Process all texts in parallel using ThreadPoolExecutor
        batch_results = [None] * len(texts)  # Pre-allocate results array
        
        with ThreadPoolExecutor(max_workers=min(len(texts), max_workers)) as executor:
            # Submit all texts for parallel processing
            futures = [
                executor.submit(_process_single_text, i, text)
                for i, text in enumerate(texts)
            ]
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(futures):
                text_index, audio = future.result()
                batch_results[text_index] = audio
                completed_count += 1
                progress_percent = int(100 * completed_count / len(texts))
                print(f"  📊 Progress: {completed_count}/{len(texts)} completed ({progress_percent}%)")
        
        print(f"✅ PARALLEL PROCESSING COMPLETED: {len(batch_results)} audio segments with FULL quality (CFG enabled)")
        return batch_results
    
