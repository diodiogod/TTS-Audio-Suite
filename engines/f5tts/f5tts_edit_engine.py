"""
F5-TTS editing engine for speech synthesis and editing.
Exact working implementation extracted from f5tts_edit_node.py
"""

import torch
import torchaudio
import tempfile
import os
from typing import List, Tuple, Optional
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
                          edit_options: Optional[dict] = None) -> torch.Tensor:
        """
        Perform F5-TTS speech editing - exact working implementation
        """
        # Set default target_rms value (not in method signature but needed internally)
        target_rms = 0.1
        
        try:
            # Import F5-TTS modules
            from f5_tts.model import CFM
            from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
            from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
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
            
            # Load checkpoint
            ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.safetensors"))
            
            # Load vocoder
            vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False)
            
            # Get tokenizer with proper error handling for missing vocab file
            try:
                vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
            except FileNotFoundError as e:
                print(f"⚠️ Global vocab file not found: {e}")
                print("📦 Attempting to use local vocab file from F5-TTS model...")
                
                # Try to use the local vocab file that we already have
                try:
                    import folder_paths
                    local_vocab_path = os.path.join(folder_paths.models_dir, "F5-TTS", "F5TTS_Base", "vocab.txt")
                    
                    if os.path.exists(local_vocab_path):
                        print(f"✅ Found local vocab file: {local_vocab_path}")
                        
                        # Load vocab manually from local file
                        with open(local_vocab_path, "r", encoding="utf-8") as f:
                            vocab_char_map = {}
                            for i, char in enumerate(f.read().strip().split('\n')):
                                vocab_char_map[char] = i
                        
                        # Check if we need to add missing tokens (model expects 2546, we have 2544)
                        vocab_size = len(vocab_char_map)
                        expected_size = 2546  # Based on the error message
                        
                        if vocab_size < expected_size:
                            print(f"⚠️ Vocab size mismatch: loaded {vocab_size}, model expects {expected_size}")
                            print("🔧 Adding missing tokens...")
                            
                            # Add common missing tokens
                            missing_tokens = ["<pad>", "<unk>"]
                            for token in missing_tokens:
                                if token not in vocab_char_map:
                                    vocab_char_map[token] = vocab_size
                                    vocab_size += 1
                                    if vocab_size >= expected_size:
                                        break
                            
                            # If still not enough, add placeholder tokens
                            while vocab_size < expected_size:
                                placeholder_token = f"<placeholder_{vocab_size}>"
                                vocab_char_map[placeholder_token] = vocab_size
                                vocab_size += 1
                        
                        print(f"✅ Final vocab size: {vocab_size} tokens")
                        
                        # Try to copy to expected location for future use (optional)
                        try:
                            import shutil
                            import site
                            
                            # Find the site-packages directory
                            site_packages = None
                            for path in site.getsitepackages():
                                if 'site-packages' in path:
                                    site_packages = path
                                    break
                            
                            if site_packages:
                                target_vocab_dir = os.path.join(site_packages, "f5_tts", "..", "..", "data", "Emilia_ZH_EN_pinyin")
                                target_vocab_dir = os.path.normpath(target_vocab_dir)
                                os.makedirs(target_vocab_dir, exist_ok=True)
                                target_vocab_path = os.path.join(target_vocab_dir, "vocab.txt")
                                
                                shutil.copy2(local_vocab_path, target_vocab_path)
                                print(f"✅ Copied local vocab to expected location: {target_vocab_path}")
                            else:
                                print("⚠️ Could not find site-packages directory, skipping vocab copy")
                        
                        except Exception as copy_error:
                            print(f"⚠️ Failed to copy vocab file (continuing anyway): {copy_error}")
                            # Don't raise error - we already have the vocab loaded successfully
                        
                    else:
                        print(f"❌ Local vocab file not found at: {local_vocab_path}")
                        raise FileNotFoundError(f"Cannot find local vocab file: {local_vocab_path}")
                        
                except Exception as local_error:
                    print(f"❌ Failed to use local vocab file: {local_error}")
                    raise FileNotFoundError(f"Cannot find or use vocab file: {e}")
            
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