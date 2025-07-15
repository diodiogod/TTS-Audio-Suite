"""
F5-TTS editing engine for speech synthesis and editing.

This module handles the core F5-TTS model loading, inference, and audio processing
for speech editing functionality.
"""

import torch
import torchaudio
import tempfile
from typing import List, Tuple, Optional
from .audio_compositing import AudioCompositor, EditMaskGenerator


class F5TTSEditEngine:
    """Core engine for F5-TTS speech editing operations."""
    
    def __init__(self, device: str, f5tts_sample_rate: int = 24000):
        """
        Initialize the F5-TTS edit engine.
        
        Args:
            device: Device to run inference on ('cuda', 'cpu', or 'auto')
            f5tts_sample_rate: Sample rate expected by F5-TTS models
        """
        self.device = self._resolve_device(device)
        self.f5tts_sample_rate = f5tts_sample_rate
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string from 'auto' to actual device"""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def perform_f5tts_edit(self, audio_tensor: torch.Tensor, sample_rate: int,
                          original_text: str, target_text: str,
                          edit_regions: List[Tuple[float, float]],
                          fix_durations: Optional[List[float]],
                          temperature: float, speed: float, target_rms: float,
                          nfe_step: int, cfg_strength: float, sway_sampling_coef: float,
                          ode_method: str) -> torch.Tensor:
        """
        Perform F5-TTS speech editing with audio compositing.
        
        Args:
            audio_tensor: Input audio tensor
            sample_rate: Audio sample rate
            original_text: Original text content
            target_text: Target text for editing
            edit_regions: List of (start_time, end_time) tuples for editing
            fix_durations: Optional fixed durations for edit regions
            temperature: Sampling temperature
            speed: Speed adjustment
            target_rms: Target RMS level
            nfe_step: Number of function evaluations
            cfg_strength: Classifier-free guidance strength
            sway_sampling_coef: Sway sampling coefficient
            ode_method: ODE solver method
            
        Returns:
            Composited audio tensor with edits applied
        """
        try:
            # Import F5-TTS modules (use original working approach)
            from f5_tts.model import CFM
            from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
            from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
            from omegaconf import OmegaConf
            from hydra.utils import get_class
            from importlib.resources import files
            from cached_path import cached_path
            import torch.nn.functional as F
            
            # Model configuration - get model name from current model or default
            model_name = "F5TTS_v1_Base"  # Use working default
            exp_name = model_name if model_name in ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"] else "F5TTS_v1_Base"
            ckpt_step = 1250000 if exp_name == "F5TTS_v1_Base" else 1200000
            
            # Load model config
            model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
            model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
            model_arc = model_cfg.model.arch
            
            dataset_name = model_cfg.datasets.name
            tokenizer = model_cfg.model.tokenizer
            
            mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
            target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
            n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
            hop_length = model_cfg.model.mel_spec.hop_length
            win_length = model_cfg.model.mel_spec.win_length
            n_fft = model_cfg.model.mel_spec.n_fft
            
            # Load checkpoint
            ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.safetensors"))
            
            # Load vocoder
            vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False)
            
            # Get tokenizer
            vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
            
            # Create model
            model = CFM(
                transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
                mel_spec_kwargs=dict(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
                odeint_kwargs=dict(
                    method=ode_method,
                ),
                vocab_char_map=vocab_char_map,
            ).to(self.device)
            
            # Load checkpoint
            load_checkpoint(model, ckpt_path, self.device)
            
            # Process audio
            audio = audio_tensor.clone()
            
            # Ensure correct tensor shape [1, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)  # Add channel dimension -> [1, samples]
            
            # Convert to mono if stereo
            if audio.dim() > 1 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != target_sample_rate:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate).to(self.device)
                audio = resampler(audio)
            
            # Normalize RMS
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            
            # Store original audio for compositing (after resampling and normalization)
            original_audio_for_compositing = audio.clone()
            
            # Create edit mask and modified audio using the modularized component
            edited_audio, edit_mask = EditMaskGenerator.create_edit_mask_and_audio(
                audio, edit_regions, fix_durations, target_sample_rate, hop_length, self.f5tts_sample_rate
            )
            
            edited_audio = edited_audio.to(self.device)
            edit_mask = edit_mask.to(self.device)
            
            # Prepare text
            text_list = [target_text]
            if tokenizer == "pinyin":
                final_text_list = convert_char_to_pinyin(text_list)
            else:
                final_text_list = text_list
            
            print(f"Original text: {original_text}")
            print(f"Target text: {target_text}")
            print(f"Edit regions: {edit_regions}")
            
            # Calculate duration
            duration = edited_audio.shape[-1] // hop_length
            
            # Validate and clamp nfe_step to prevent ODE solver issues
            safe_nfe_step = max(1, min(nfe_step, 71))
            if safe_nfe_step != nfe_step:
                print(f"⚠️ F5-TTS Edit: Clamped nfe_step from {nfe_step} to {safe_nfe_step} to prevent ODE solver issues")
            
            # Perform inference
            with torch.inference_mode():
                generated, trajectory = model.sample(
                    cond=edited_audio,
                    text=final_text_list,
                    duration=duration,
                    steps=safe_nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    seed=None,  # Will be set by the model if needed
                    edit_mask=edit_mask,
                )
                
                print(f"Generated mel: {generated.shape}")
                
                # Generate final audio
                generated = generated.to(torch.float32)
                gen_mel_spec = generated.permute(0, 2, 1)
                
                if mel_spec_type == "vocos":
                    generated_wave = vocoder.decode(gen_mel_spec).cpu()
                elif mel_spec_type == "bigvgan":
                    generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
                else:
                    generated_wave = vocoder(gen_mel_spec).cpu()
                
                # Apply RMS correction
                if rms < target_rms:
                    # Ensure all tensors are on the same device (CPU) for RMS correction
                    rms_cpu = rms.cpu() if hasattr(rms, 'device') else rms
                    target_rms_cpu = target_rms.cpu() if hasattr(target_rms, 'device') else target_rms
                    generated_wave = generated_wave * rms_cpu / target_rms_cpu
                
                print(f"Generated wave: {generated_wave.shape}")
                
                # Composite the edited audio with original audio to preserve quality outside edit regions
                composite_audio = AudioCompositor.composite_edited_audio(
                    original_audio_for_compositing.cpu(),
                    generated_wave,
                    edit_regions,
                    target_sample_rate
                )
                
                print(f"Composite audio: {composite_audio.shape}")
                
                return composite_audio
                
        except ImportError as e:
            raise ImportError(f"F5-TTS modules not available for speech editing: {e}")
        except Exception as e:
            raise RuntimeError(f"F5-TTS speech editing failed: {e}")
    
    @staticmethod
    def save_audio_temp(audio: torch.Tensor, sample_rate: int) -> str:
        """
        Save audio tensor to temporary file.
        
        Args:
            audio: Audio tensor to save
            sample_rate: Sample rate for the audio
            
        Returns:
            Path to the temporary audio file
        """
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Ensure audio is in correct format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(temp_path, audio, sample_rate)
        return temp_path