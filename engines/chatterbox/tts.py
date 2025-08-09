from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import warnings
# Import safetensors for multilanguage model support
from safetensors.torch import load_file

# Import folder_paths for model directory detection
try:
    import folder_paths
except ImportError:
    folder_paths = None

# Import perth with warnings disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import perth

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# Import language model registry
try:
    from .language_models import get_model_config, CHATTERBOX_MODELS
except ImportError:
    # Fallback if language_models not available
    CHATTERBOX_MODELS = {"English": {"repo": "ResembleAI/chatterbox", "format": "pt"}}
    def get_model_config(language):
        return CHATTERBOX_MODELS.get(language, CHATTERBOX_MODELS["English"])

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


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # Initialize watermarker silently (but disabled by default)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.watermarker = perth.PerthImplicitWatermarker()
        self.enable_watermarking = False  # Disabled by default for maximum compatibility

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        print(f"📦 Loading local ChatterBox models from: {ckpt_dir}")
        ckpt_dir = Path(ckpt_dir)
        
        # Auto-detect model format
        def load_model_file(base_name: str):
            """Load model file with auto-detection of format (.safetensors preferred over .pt)"""
            safetensors_path = ckpt_dir / f"{base_name}.safetensors"
            pt_path = ckpt_dir / f"{base_name}.pt"
            
            if safetensors_path.exists():
                print(f"📁 Loading {base_name} from safetensors format")
                return load_file(safetensors_path, device=device)
            elif pt_path.exists():
                print(f"📁 Loading {base_name} from pt format")
                return torch.load(pt_path, map_location=device)
            else:
                raise FileNotFoundError(f"Neither {base_name}.safetensors nor {base_name}.pt found in {ckpt_dir}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load VoiceEncoder
            ve = VoiceEncoder()
            ve_state = load_model_file("ve")
            ve.load_state_dict(ve_state)
            ve.to(device).eval()

            # Load T3 config
            t3_state = load_model_file("t3_cfg")
            if "model" in t3_state.keys():
                t3_state = t3_state["model"][0]
            
            # Create config with proper settings
            from .models.t3.t3 import T3Config
            config = T3Config()
            
            # Initialize model with config
            t3 = T3(config)
            
            # Load state and ensure settings
            t3.load_state_dict(t3_state)
            t3.tfmr.output_attentions = False
            
            t3.to(device).eval()

            # Load S3Gen
            s3gen = S3Gen()
            s3gen_state = load_model_file("s3gen")
            # Apply JaneDoe84's critical fix: strict=False to handle missing keys
            s3gen.load_state_dict(s3gen_state, strict=False)
            s3gen.to(device).eval()

            tokenizer = EnTokenizer(
                str(ckpt_dir / "tokenizer.json")
            )

            conds = None
            if (builtin_voice := ckpt_dir / "conds.pt").exists():
                conds = Conditionals.load(builtin_voice).to(device)

            instance = cls(t3, s3gen, ve, tokenizer, device, conds=conds)
            print("✅ Successfully loaded all local ChatterBox models")
            return instance

    @classmethod
    def from_pretrained(cls, device, language="English") -> 'ChatterboxTTS':
        """
        Load ChatterBox model from HuggingFace Hub with language support.
        
        Args:
            device: Device to load model on
            language: Language model to load (English, German, Norwegian, etc.)
        """
        # Get model configuration for the specified language
        model_config = get_model_config(language)
        if not model_config:
            print(f"⚠️ Language '{language}' not found, falling back to English")
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
                        local_loader=lambda path: cls.from_local(path, device),
                        fallback_loader=lambda: cls.from_pretrained(device, language="English"),
                        fallback_name="English",
                        original_request=language
                    )
                except Exception as e:
                    print(f"⚠️ Fallback failed: {e}, proceeding with direct HuggingFace download")
            
            model_config = get_model_config("English")
        
        repo_id = model_config.get("repo", REPO_ID)
        model_format = model_config.get("format", "pt")
        
        print(f"📦 Loading ChatterBox model for {language} from {repo_id}")
        
        # Define file extensions based on format
        if model_format == "safetensors":
            file_extensions = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
        elif model_format == "pt":
            file_extensions = ["ve.pt", "t3_cfg.pt", "s3gen.pt", "tokenizer.json", "conds.pt"]
        else:
            # Auto format - try both, safetensors preferred
            print("🔍 Auto-detecting model format...")
            file_extensions = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
        
        # Download files
        local_paths = []
        for fpath in file_extensions:
            try:
                local_path = hf_hub_download(repo_id=repo_id, filename=fpath)
                local_paths.append(local_path)
            except Exception as e:
                # If safetensors fails, try .pt format
                if fpath.endswith('.safetensors'):
                    fallback_fpath = fpath.replace('.safetensors', '.pt')
                    print(f"⚠️ {fpath} not found, trying {fallback_fpath}")
                    try:
                        local_path = hf_hub_download(repo_id=repo_id, filename=fallback_fpath)
                        local_paths.append(local_path)
                    except Exception as e2:
                        print(f"❌ Failed to download both {fpath} and {fallback_fpath}: {e2}")
                        raise e2
                else:
                    print(f"❌ Failed to download {fpath}: {e}")
                    raise e

        # Use the directory of the first downloaded file
        model_dir = Path(local_paths[0]).parent
        return cls.from_local(model_dir, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

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
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        """Generate audio for a single text input."""
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
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
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                return torch.from_numpy(watermarked_wav).unsqueeze(0)
            else:
                return torch.from_numpy(wav).unsqueeze(0)
    
    def generate_batch(
        self,
        texts,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        batch_size=4,
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
        
        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        
        print(f"🚀 ChatterBox TRUE BATCH processing: {len(texts)} texts")
        
        results = []
        
        # Process texts in true batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"⚡ Processing batch {i//batch_size + 1}: {len(batch_texts)} texts")
            
            try:
                # NEW: True batch processing with simultaneous inference
                batch_audio = self._batch_inference_simultaneous(
                    batch_texts, temperature, cfg_weight
                )
                results.extend(batch_audio)
                print(f"✅ Batch {i//batch_size + 1} completed successfully")
                
            except Exception as e:
                print(f"⚠️ Batch inference failed: {e}")
                print(f"🔄 Falling back to individual processing for this batch")
                
                # Fallback to individual processing for this batch only
                for text in batch_texts:
                    try:
                        individual_audio = self.generate(
                            text=text,
                            audio_prompt_path=None,  # Use already loaded conditionals
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                            apply_watermark=self.enable_watermarking,
                        )
                        results.append(individual_audio)
                    except Exception as individual_error:
                        print(f"❌ Individual generation also failed for text: {text[:50]}...")
                        print(f"Error: {individual_error}")
                        # Return empty tensor as placeholder
                        results.append(torch.zeros(1, 1000))
        
        return results
    
    def _batch_inference_simultaneous(self, texts, temperature=0.8, cfg_weight=0.5):
        """
        NEW METHOD: Perform true simultaneous batch inference.
        This is the core improvement that enables real batch processing.
        """
        # Prepare all text tokens at once
        all_text_tokens = []
        for text in texts:
            normed_text = punc_norm(text)
            text_tokens = self.tokenizer.text_to_tokens(normed_text).to(self.device)
            
            if cfg_weight > 0.0:
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG duplication
            
            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            all_text_tokens.append(text_tokens)
        
        # Find max length and pad all sequences
        max_len = max(tokens.shape[-1] for tokens in all_text_tokens)
        padded_tokens = []
        
        for tokens in all_text_tokens:
            if tokens.shape[-1] < max_len:
                pad_amount = max_len - tokens.shape[-1]
                tokens = F.pad(tokens, (0, pad_amount), value=self.t3.hp.stop_text_token)
            padded_tokens.append(tokens)
        
        # Stack into true batch tensor
        batch_text_tokens = torch.stack(padded_tokens, dim=0)  # [batch_size, seq_len] or [batch_size*2, seq_len] for CFG
        
        # Replicate T3 conditioning for batch
        batch_t3_cond = self._replicate_conditioning_for_batch(len(texts), cfg_weight > 0.0)
        
        print(f"🔥 T3 batch inference: {batch_text_tokens.shape[0]} sequences")
        
        with torch.inference_mode():
            # TRUE BATCH INFERENCE: All texts processed simultaneously
            batch_speech_tokens = self.t3.batch_inference(
                t3_cond=batch_t3_cond,
                text_tokens=batch_text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            
            # Extract results and generate audio
            batch_results = []
            effective_batch_size = len(texts)
            
            for i in range(effective_batch_size):
                if cfg_weight > 0.0:
                    # Extract conditional result (first half)
                    speech_tokens = batch_speech_tokens[i]
                else:
                    speech_tokens = batch_speech_tokens[i]
                
                # Process tokens
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens.to(self.device)
                
                # Generate audio
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()
                
                if self.enable_watermarking:
                    watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                    batch_results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))
                else:
                    batch_results.append(torch.from_numpy(wav).unsqueeze(0))
            
            return batch_results
    
    def _replicate_conditioning_for_batch(self, batch_size, use_cfg):
        """
        Replicate T3 conditioning for batch processing.
        """
        original_cond = self.conds.t3
        
        # Determine effective batch size (doubled for CFG)
        effective_batch_size = batch_size * 2 if use_cfg else batch_size
        
        # Replicate conditioning tensors
        speaker_emb = original_cond.speaker_emb.repeat(effective_batch_size, 1)
        
        cond_prompt_speech_tokens = None
        if original_cond.cond_prompt_speech_tokens is not None:
            cond_prompt_speech_tokens = original_cond.cond_prompt_speech_tokens.repeat(effective_batch_size, 1)
        
        emotion_adv = original_cond.emotion_adv.repeat(effective_batch_size, 1, 1)
        
        return T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=cond_prompt_speech_tokens,
            emotion_adv=emotion_adv,
        ).to(device=self.device)
