"""
Stateless ChatterBox Wrapper for Thread-Safe Parallel Processing

This wrapper eliminates shared state corruption by calculating conditions fresh 
for each generation call instead of storing them in self.conds.

Key benefits:
- Thread-safe parallel generation without locks
- No shared state corruption
- True parallelism with multiple workers
- Same memory footprint (single model instance)
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import warnings

# Use librosa fallback for Python 3.13 compatibility
from utils.audio.librosa_fallback import safe_load, safe_resample

# Import ChatterBox components
from engines.chatterbox.models.s3tokenizer import S3_SR, drop_invalid_tokens
from engines.chatterbox.models.s3gen import S3GEN_SR
from engines.chatterbox.models.t3.modules.cond_enc import T3Cond
from engines.chatterbox.tts import Conditionals, punc_norm


class StatelessChatterBoxWrapper:
    """
    Thread-safe stateless wrapper for ChatterBox TTS model.
    
    This wrapper ensures that each generation call is completely independent,
    with no shared state that could be corrupted by concurrent access.
    All conditions are calculated fresh for each call and passed as parameters.
    """
    
    def __init__(self, chatterbox_model):
        """
        Initialize the stateless wrapper.
        
        Args:
            chatterbox_model: Existing ChatterBox TTS model instance
        """
        self.model = chatterbox_model
        
        # Store model components for direct access
        self.t3 = chatterbox_model.t3
        self.s3gen = chatterbox_model.s3gen
        self.tokenizer = chatterbox_model.tokenizer
        self.ve = chatterbox_model.ve
        self.device = chatterbox_model.device
        self.sr = chatterbox_model.sr
        
        # Store constants
        self.ENC_COND_LEN = chatterbox_model.ENC_COND_LEN
        self.DEC_COND_LEN = chatterbox_model.DEC_COND_LEN
        
        # Watermarking settings
        self.enable_watermarking = getattr(chatterbox_model, 'enable_watermarking', False)
        self.watermarker = getattr(chatterbox_model, 'watermarker', None)
        
        # IMPORTANT: We do NOT store self.conds - that's the whole point!
        
    def generate_stateless(self, text, audio_prompt_path=None, exaggeration=0.5, 
                          cfg_weight=0.5, temperature=0.8, seed=None):
        """
        Generate audio without modifying any shared state.
        
        This method is completely thread-safe and can be called concurrently
        by multiple workers without any risk of state corruption.
        
        Args:
            text: Text to generate speech for
            audio_prompt_path: Path to reference audio file (optional)
            exaggeration: Emotion exaggeration factor (0.0 to 1.0+)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            seed: Random seed for reproducibility (optional)
            
        Returns:
            torch.Tensor: Generated audio waveform
        """
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 1. Prepare conditions locally (no state modification)
        local_conds = self._prepare_conditions_locally(audio_prompt_path, exaggeration)
        
        # 2. Normalize and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG
        
        # Add start/end tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        # 3. Generate speech tokens using local conditions (thread-safe)
        with torch.no_grad():  # Use no_grad instead of inference_mode for better compatibility
            speech_tokens = self.t3.inference(
                t3_cond=local_conds.t3,  # Fresh, isolated conditions
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            
            # Extract only the conditional batch and clone to avoid inference tensor issues
            speech_tokens = speech_tokens[0].detach().clone()
            
            # Drop invalid tokens
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            
            # 4. Generate waveform using local conditions
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=local_conds.gen,  # Fresh, isolated conditions
            )
            
            # Ensure wav is a regular tensor
            if hasattr(wav, 'detach'):
                wav = wav.detach().clone()
            wav = wav.squeeze(0).cpu().numpy()
            
            # 5. Apply watermarking if enabled
            if self.enable_watermarking and self.watermarker:
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                # Return as regular tensor (not inference tensor)
                return torch.from_numpy(watermarked_wav).unsqueeze(0).detach()
            else:
                # Return as regular tensor (not inference tensor)
                return torch.from_numpy(wav).unsqueeze(0).detach()
    
    def _prepare_conditions_locally(self, wav_fpath, exaggeration=0.5):
        """
        Prepare conditions without storing them in self.
        
        This is a stateless version of the original prepare_conditionals() method.
        It calculates all necessary conditions and returns them without modifying
        any instance state.
        
        Args:
            wav_fpath: Path to reference audio file (can be None)
            exaggeration: Emotion exaggeration factor
            
        Returns:
            Conditionals: Fresh conditions object for this generation
        """
        # Ensure we're not in any autograd context for the entire preparation
        with torch.no_grad():
            # If no audio prompt, create default conditions
            if not wav_fpath:
                return self._get_default_conditions(exaggeration)
            
            # Load reference wav using fallback for Python 3.13 compatibility
            s3gen_ref_wav, sample_rate = safe_load(wav_fpath, sr=S3GEN_SR, mono=True)
            
            # Resample to 16k for S3 tokenizer using fallback
            ref_16k_wav = safe_resample(s3gen_ref_wav, S3GEN_SR, S3_SR)
            
            # Prepare S3Gen reference
            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
            
            # Speech cond prompt tokens
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device).detach().clone()
            
            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device).detach().clone()
            
            # Create T3 conditions
            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, device=self.device).detach(),
            ).to(device=self.device)
            
            # Clone ref_dict tensors to avoid shared references
            cloned_ref_dict = {}
            for k, v in s3gen_ref_dict.items():
                if torch.is_tensor(v):
                    cloned_ref_dict[k] = v.detach().clone()
                else:
                    cloned_ref_dict[k] = v
            
            # Return fresh Conditionals object (not stored!)
            return Conditionals(t3_cond, cloned_ref_dict)
    
    def _get_default_conditions(self, exaggeration=0.5):
        """
        Get default conditions when no reference audio is provided.
        
        Creates minimal valid conditions for generation without reference audio.
        This is used when audio_prompt_path is None.
        
        Args:
            exaggeration: Emotion exaggeration factor
            
        Returns:
            Conditionals: Default conditions object
        """
        with torch.no_grad():
            # Check if model has default conditions
            if hasattr(self.model, 'conds') and self.model.conds is not None:
                # Create a copy of existing conditions with updated exaggeration
                existing_conds = self.model.conds
            
                # Create new T3Cond with updated exaggeration
                t3_cond = T3Cond(
                    speaker_emb=existing_conds.t3.speaker_emb.clone().detach() if existing_conds.t3.speaker_emb is not None else None,
                    cond_prompt_speech_tokens=existing_conds.t3.cond_prompt_speech_tokens.clone().detach() if existing_conds.t3.cond_prompt_speech_tokens is not None else None,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1, device=self.device).detach(),
                ).to(device=self.device)
                
                # Clone gen dict
                gen_dict = {}
                for k, v in existing_conds.gen.items():
                    if torch.is_tensor(v):
                        gen_dict[k] = v.clone().detach()
                    else:
                        gen_dict[k] = v
                
                return Conditionals(t3_cond, gen_dict)
            
            # If no existing conditions, raise error (user must provide audio_prompt_path)
            raise ValueError("No reference audio provided and no default conditions available. "
                            "Please provide audio_prompt_path or call prepare_conditionals() first on the base model.")
    
    def prepare_default_conditionals(self, wav_fpath, exaggeration=0.5):
        """
        Prepare and store default conditions in the base model.
        
        This method can be called once to set up default conditions that will be
        used when no audio_prompt_path is provided. It modifies the base model's
        state, so it should only be called during initialization, not during
        parallel processing.
        
        Args:
            wav_fpath: Path to reference audio file
            exaggeration: Default emotion exaggeration factor
        """
        # Use the base model's prepare_conditionals method
        self.model.prepare_conditionals(wav_fpath, exaggeration)
        print(f"✅ Default conditions prepared for stateless wrapper")
    
    @property
    def has_default_conditions(self):
        """Check if the wrapper has default conditions available."""
        return hasattr(self.model, 'conds') and self.model.conds is not None
    
    def __repr__(self):
        """String representation of the wrapper."""
        return f"StatelessChatterBoxWrapper(device={self.device}, has_defaults={self.has_default_conditions})"