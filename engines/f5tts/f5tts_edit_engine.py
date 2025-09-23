"""
F5-TTS editing engine for speech synthesis and editing.
Exact working implementation extracted from f5tts_edit_node.py
"""

import torch
import torchaudio
import tempfile
import os
from typing import List, Tuple, Optional, Any
from .audio_compositing import AudioCompositor, EditMaskGenerator


class F5TTSEditEngine:
    """Core engine for F5-TTS speech editing operations."""
    
    def __init__(self, device: str, f5tts_sample_rate: int = 24000):
        """Initialize the F5-TTS edit engine."""
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
                          temperature: float,
                          nfe_step: int, cfg_strength: float, sway_sampling_coef: float,
                          ode_method: str, seed: int, current_model_name: str = "F5TTS_v1_Base",
                          edit_options: Optional[dict] = None, 
                          unified_model: Optional[Any] = None) -> torch.Tensor:
        """
        Perform F5-TTS speech editing - exact working implementation
        """
        # Set default target_rms value (not in method signature but needed internally)
        target_rms = 0.1
        
        try:
            # Import all needed functions at the top to avoid scoping issues
            from engines.f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
            from importlib.resources import files
            from omegaconf import OmegaConf
            import torch.nn.functional as F
            
            # Use unified model if provided, otherwise fall back to direct loading
            if unified_model is not None:
                # Extract the F5-TTS model from the unified wrapper
                if hasattr(unified_model, 'f5tts_model'):
                    f5tts_api = unified_model.f5tts_model
                elif hasattr(unified_model, 'model'):
                    f5tts_api = unified_model.model
                else:
                    f5tts_api = unified_model
                
                # Get model components from the unified F5-TTS instance
                model = f5tts_api.ema_model
                vocoder = f5tts_api.vocoder
                target_sample_rate = f5tts_api.target_sample_rate
                mel_spec_type = f5tts_api.mel_spec_type
                
                # Get config parameters - use standard E2TTS/F5TTS defaults since unified model doesn't expose these
                # These are standard parameters for E2TTS_Base and F5TTS models
                hop_length = 256
                win_length = 1024
                n_fft = 1024
                n_mel_channels = 100
                tokenizer = "pinyin"  # Default tokenizer for E2TTS and F5TTS
                
                # For E2TTS, we can also load the config to get exact values
                try:
                    # Determine model config based on current model name  
                    exp_name = current_model_name if current_model_name in ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"] else "F5TTS_v1_Base"
                    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
                    
                    # Override with actual config values
                    hop_length = model_cfg.model.mel_spec.hop_length
                    win_length = model_cfg.model.mel_spec.win_length
                    n_fft = model_cfg.model.mel_spec.n_fft
                    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
                    tokenizer = model_cfg.model.tokenizer  # Also get tokenizer from config
                except Exception as config_error:
                    print(f"⚠️ Could not load model config, using defaults: {config_error}")
                    # Keep the default values set above
                
                print("✅ Using unified model for F5-TTS speech editing")
                
            else:
                # Fallback to direct model loading (legacy behavior)
                print("⚠️ No unified model provided, falling back to direct loading")
                
                # Import F5-TTS modules
                from engines.f5_tts.model import CFM
                from engines.f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
                from engines.f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
                from omegaconf import OmegaConf
                from hydra.utils import get_class
                from importlib.resources import files
                from cached_path import cached_path
                import torch.nn.functional as F
                
                # Model configuration - get model name from current model or default
                model_name = current_model_name
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
                
                # Load checkpoint - handle E2TTS vs F5TTS repository mapping
                repo_name = "F5-TTS"  # Default repository
                if exp_name == "E2TTS_Base":
                    repo_name = "E2-TTS"  # E2TTS models are in E2-TTS repository
                
                ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
                
                # Load vocoder
                vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False)
                
                # Get tokenizer using vocab file path like the working F5TTS API
                # First try to get vocab file path from local model
                vocab_file_path = None
                import folder_paths
                
                # Try TTS path first, then legacy paths  
                vocab_search_paths = [
                    os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", "F5TTS_Base", "vocab.txt"),
                    os.path.join(folder_paths.models_dir, "F5-TTS", "F5TTS_Base", "vocab.txt"),  # Legacy
                    os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS", "F5TTS_Base", "vocab.txt")  # Legacy
                ]
                
                for path in vocab_search_paths:
                    if os.path.exists(path):
                        vocab_file_path = path
                        break
                
                # Try get_tokenizer with vocab file path like working F5TTS API does
                try:
                    if vocab_file_path:
                        vocab_char_map, vocab_size = get_tokenizer(vocab_file_path, "custom")
                    else:
                        # Fallback to original dataset name approach
                        vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
                except FileNotFoundError as e:
                    print(f"⚠️ Vocab file not found: {e}")
                    # Use the fallback approach like F5TTS API - let the library handle it with default vocab
                    from importlib.resources import files
                    default_vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
                    print(f"📦 Using F5-TTS default vocab: {default_vocab_file}")
                    vocab_char_map, vocab_size = get_tokenizer(default_vocab_file, "custom")
                
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
                dtype = torch.float32 if mel_spec_type == "bigvgan" else None
                model = load_checkpoint(model, ckpt_path, self.device, dtype=dtype, use_ema=True)
            
            # Prepare audio - ensure consistent dimensions
            audio = audio_tensor.to(self.device)
            
            # Handle different input formats - ensure we have 2D tensor [channels, samples]
            if audio.dim() == 3:  # [batch, channels, samples]
                audio = audio.squeeze(0)  # Remove batch dimension -> [channels, samples]
            elif audio.dim() == 1:  # [samples]
                audio = audio.unsqueeze(0)  # Add channel dimension -> [1, samples]
            
            # Convert to mono if stereo
            if audio.dim() > 1 and audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate).to(self.device)
                audio = resampler(audio)
            
            # Normalize RMS
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
            
            # Store original audio for compositing (after resampling and normalization)
            original_audio_for_compositing = audio.clone()
            
            # Create edit mask and modified audio
            edited_audio, edit_mask = EditMaskGenerator.create_edit_mask_and_audio(
                audio, edit_regions, fix_durations, target_sample_rate, hop_length, self.f5tts_sample_rate
            )
            
            edited_audio = edited_audio.to(self.device)
            edit_mask = edit_mask.to(self.device)
            
            # Prepare text
            text_list = [target_text]
            if tokenizer == "pinyin" and not (hasattr(unified_model, 'model_name') and unified_model.model_name == "F5-JP"):
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
                    seed=seed,  # Use provided seed
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
                
                # Calculate actual edit regions in generated audio (accounting for fixed durations)
                actual_edit_regions = []
                generated_time_offset = 0  # Running offset in generated audio
                original_time_offset = 0   # Running offset in original audio
                
                for i, (start, end) in enumerate(edit_regions):
                    # Get the actual duration used for this region
                    if fix_durations and i < len(fix_durations):
                        actual_duration = fix_durations[i]
                    else:
                        actual_duration = end - start
                    
                    # Add preserved audio before this edit region
                    preserved_duration = start - original_time_offset
                    generated_time_offset += preserved_duration
                    
                    # This edit region in generated audio
                    region_start = generated_time_offset
                    region_end = generated_time_offset + actual_duration
                    actual_edit_regions.append((region_start, region_end))
                    
                    # Update offsets
                    generated_time_offset += actual_duration
                    original_time_offset = end
                
                # Composite the edited audio with original audio to preserve quality outside edit regions
                composite_audio = self._build_composite_audio(
                    original_audio_for_compositing.cpu(),
                    generated_wave,
                    edit_regions,
                    actual_edit_regions,
                    target_sample_rate
                )
                
                print(f"Composite audio: {composite_audio.shape}")
                
                return composite_audio
                
        except ImportError as e:
            raise ImportError(f"F5-TTS modules not available for speech editing: {e}")
        except Exception as e:
            raise RuntimeError(f"F5-TTS speech editing failed: {e}")
    
    def _build_composite_audio(self, original_audio: torch.Tensor, generated_audio: torch.Tensor,
                              original_edit_regions: List[Tuple[float, float]], 
                              actual_edit_regions: List[Tuple[float, float]],
                              sample_rate: int) -> torch.Tensor:
        """Build composite audio using pre-calculated actual edit regions"""
        
        # Ensure both audios are mono
        if original_audio.dim() > 1:
            original_audio = torch.mean(original_audio, dim=0, keepdim=True)
        if generated_audio.dim() > 1:
            generated_audio = torch.mean(generated_audio, dim=0, keepdim=True)
        
        composite_segments = []
        original_pos = 0.0  # Current position in original audio
        
        print(f"🔨 Building composite from original: {original_audio.shape}, generated: {generated_audio.shape}")
        
        for i, ((orig_start, orig_end), (gen_start, gen_end)) in enumerate(zip(original_edit_regions, actual_edit_regions)):
            print(f"\\n🔧 Processing edit region {i}: orig({orig_start:.2f}-{orig_end:.2f}s) -> gen({gen_start:.2f}-{gen_end:.2f}s)")
            
            # Add preserved audio before this edit region (if any)
            if orig_start > original_pos:
                preserved_start_sample = int(original_pos * sample_rate)
                preserved_end_sample = int(orig_start * sample_rate)
                preserved_end_sample = min(preserved_end_sample, original_audio.shape[-1])
                
                if preserved_start_sample < preserved_end_sample:
                    preserved_segment = original_audio[:, preserved_start_sample:preserved_end_sample]
                    composite_segments.append(preserved_segment)
                    print(f"  ✅ Added preserved segment: original {original_pos:.2f}-{orig_start:.2f}s ({preserved_segment.shape[-1]} samples)")
            
            # Add edited segment from generated audio
            edit_start_sample = int(gen_start * sample_rate)
            edit_end_sample = int(gen_end * sample_rate)
            edit_end_sample = min(edit_end_sample, generated_audio.shape[-1])
            
            if edit_start_sample < edit_end_sample:
                edited_segment = generated_audio[:, edit_start_sample:edit_end_sample]
                composite_segments.append(edited_segment)
                print(f"  🎵 Added edited segment: generated {gen_start:.2f}-{gen_end:.2f}s ({edited_segment.shape[-1]} samples)")
            
            # Update position to end of original edit region
            original_pos = orig_end
        
        # Add remaining original audio after last edit region
        original_duration = original_audio.shape[-1] / sample_rate
        if original_pos < original_duration:
            remaining_start_sample = int(original_pos * sample_rate)
            remaining_segment = original_audio[:, remaining_start_sample:]
            composite_segments.append(remaining_segment)
            remaining_duration = original_duration - original_pos
            print(f"  ✅ Added remaining segment: original {original_pos:.2f}-{original_duration:.2f}s ({remaining_segment.shape[-1]} samples)")
        
        # Concatenate all segments
        if composite_segments:
            composite_audio = torch.cat(composite_segments, dim=-1)
            total_duration = composite_audio.shape[-1] / sample_rate
            print(f"🎉 Final composite: {composite_audio.shape} ({total_duration:.2f}s)")
            return composite_audio
        else:
            return generated_audio
    
    @staticmethod
    def save_audio_temp(audio: torch.Tensor, sample_rate: int) -> str:
        """Save audio tensor to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Ensure audio is in correct format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(temp_path, audio, sample_rate)
        return temp_path