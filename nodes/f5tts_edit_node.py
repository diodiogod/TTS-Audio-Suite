"""
F5-TTS Edit Node - Speech editing functionality
Enhanced Speech editing node using F5-TTS for targeted word/phrase replacement
"""

import torch
import numpy as np
import os
import tempfile
import torchaudio
from typing import Dict, Any, Optional, List, Tuple

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(current_dir, "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from core.audio_processing import AudioProcessingUtils
import comfy.model_management as model_management


class F5TTSEditNode(BaseF5TTSNode):
    """
    F5-TTS Speech editing node for targeted word/phrase replacement.
    Allows editing specific words/phrases in existing speech while maintaining voice characteristics.
    """
    
    @classmethod
    def NAME(cls):
        return "🎛️ F5-TTS Speech Editor"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_audio": ("AUDIO", {
                    "tooltip": "Original audio to edit"
                }),
                "original_text": ("STRING", {
                    "multiline": True,
                    "default": "Some call me nature, others call me mother nature.",
                    "tooltip": "Original text that matches the original audio"
                }),
                "target_text": ("STRING", {
                    "multiline": True,
                    "default": "Some call me optimist, others call me realist.",
                    "tooltip": "Target text with desired changes"
                }),
                "edit_regions": ("STRING", {
                    "multiline": True,
                    "default": "1.42,2.44\n4.04,4.9",
                    "tooltip": "Edit regions as 'start,end' in seconds (one per line). These are the time regions to replace."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run F5-TTS model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "model": (["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"], {
                    "default": "F5TTS_v1_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
            },
            "optional": {
                "fix_durations": ("STRING", {
                    "multiline": True,
                    "default": "1.2\n1.0",
                    "tooltip": "Fixed durations for each edit region in seconds (one per line). Leave empty to use original durations."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness in F5-TTS generation. Higher values = more creative/varied speech, lower values = more consistent/predictable speech."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "F5-TTS native speech speed control. 1.0 = normal speed, 0.5 = half speed (slower), 2.0 = double speed (faster)."
                }),
                "target_rms": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Target audio volume level (Root Mean Square). Controls output loudness normalization. Higher values = louder audio output."
                }),
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 71,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance. Values above 71 may cause ODE solver issues."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Classifier-Free Guidance strength. Controls how strictly F5-TTS follows the reference text. Higher values = more adherence to reference, lower values = more creative freedom."
                }),
                "sway_sampling_coef": ("FLOAT", {
                    "default": -1.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Sway sampling coefficient for F5-TTS inference. Controls the sampling behavior during generation. Negative values typically work better."
                }),
                "ode_method": (["euler", "midpoint"], {
                    "default": "euler",
                    "tooltip": "ODE solver method for F5-TTS inference. 'euler' is faster and typically sufficient, 'midpoint' may provide higher quality but slower generation."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("edited_audio", "edit_info")
    FUNCTION = "edit_speech"
    CATEGORY = "F5-TTS Voice"

    def __init__(self):
        super().__init__()
        self.current_model_name = "F5TTS_v1_Base"  # Default model name
    
    def _parse_edit_regions(self, edit_regions_str: str) -> List[Tuple[float, float]]:
        """Parse edit regions from string format"""
        regions = []
        lines = edit_regions_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                try:
                    start, end = map(float, line.split(','))
                    regions.append((start, end))
                except ValueError:
                    raise ValueError(f"Invalid edit region format: '{line}'. Expected 'start,end' format.")
        return regions
    
    def _parse_fix_durations(self, fix_durations_str: str) -> Optional[List[float]]:
        """Parse fix durations from string format"""
        if not fix_durations_str.strip():
            return None
        
        durations = []
        lines = fix_durations_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                try:
                    duration = float(line)
                    durations.append(duration)
                except ValueError:
                    raise ValueError(f"Invalid fix duration format: '{line}'. Expected a number.")
        return durations
    
    def _create_edit_mask_and_audio(self, original_audio: torch.Tensor, edit_regions: List[Tuple[float, float]], 
                                   fix_durations: Optional[List[float]], target_sample_rate: int, 
                                   hop_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edit mask and modified audio with silence gaps for editing"""
        if original_audio.dim() > 1:
            original_audio = torch.mean(original_audio, dim=0, keepdim=True)
        
        # Ensure we have the right sample rate
        if target_sample_rate != self.f5tts_sample_rate:
            print(f"⚠️ Resampling audio from {target_sample_rate}Hz to {self.f5tts_sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(target_sample_rate, self.f5tts_sample_rate)
            original_audio = resampler(original_audio)
            target_sample_rate = self.f5tts_sample_rate
        
        # Normalize audio level
        rms = torch.sqrt(torch.mean(torch.square(original_audio)))
        if rms < 0.1:
            original_audio = original_audio * 0.1 / rms
        
        # Build edited audio and mask (ensure they're on the same device as input audio)
        device = original_audio.device
        offset = 0
        edited_audio = torch.zeros(1, 0, device=device)
        edit_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        
        fix_durations_copy = fix_durations.copy() if fix_durations else None
        
        for i, (start, end) in enumerate(edit_regions):
            # Get duration for this edit region
            if fix_durations_copy:
                part_dur = fix_durations_copy.pop(0) if fix_durations_copy else (end - start)
            else:
                part_dur = end - start
            
            # Convert to samples
            part_dur_samples = int(part_dur * target_sample_rate)
            start_samples = int(start * target_sample_rate)
            end_samples = int(end * target_sample_rate)
            offset_samples = int(offset * target_sample_rate)
            
            # Add audio before edit region (preserve original)
            pre_edit_audio = original_audio[:, offset_samples:start_samples]
            silence_tensor = torch.zeros(1, part_dur_samples, device=device)
            edited_audio = torch.cat((edited_audio, pre_edit_audio, silence_tensor), dim=-1)
            
            # Add mask - True for preserved audio, False for edited regions
            pre_edit_mask_length = int((start_samples - offset_samples) / hop_length)
            edit_mask_length = int(part_dur_samples / hop_length)
            
            edit_mask = torch.cat((
                edit_mask,
                torch.ones(1, pre_edit_mask_length, dtype=torch.bool, device=device),
                torch.zeros(1, edit_mask_length, dtype=torch.bool, device=device)
            ), dim=-1)
            
            offset = end
        
        # Add remaining audio after last edit region
        remaining_samples = int(offset * target_sample_rate)
        if remaining_samples < original_audio.shape[-1]:
            remaining_audio = original_audio[:, remaining_samples:]
            edited_audio = torch.cat((edited_audio, remaining_audio), dim=-1)
            
            remaining_mask_length = int(remaining_audio.shape[-1] / hop_length)
            edit_mask = torch.cat((edit_mask, torch.ones(1, remaining_mask_length, dtype=torch.bool, device=device)), dim=-1)
        
        # Pad mask to match audio length
        required_mask_length = edited_audio.shape[-1] // hop_length + 1
        current_mask_length = edit_mask.shape[-1]
        if current_mask_length < required_mask_length:
            padding = required_mask_length - current_mask_length
            edit_mask = torch.cat((edit_mask, torch.ones(1, padding, dtype=torch.bool, device=device)), dim=-1)
        
        return edited_audio, edit_mask
    
    def _save_audio_temp(self, audio: torch.Tensor, sample_rate: int) -> str:
        """Save audio tensor to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Ensure audio is in correct format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        torchaudio.save(temp_path, audio, sample_rate)
        return temp_path
    
    def edit_speech(self, original_audio, original_text, target_text, edit_regions, 
                   device, model, seed, fix_durations="", temperature=0.8, speed=1.0, 
                   target_rms=0.1, nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, 
                   ode_method="euler"):
        
        def _process():
            # Validate inputs
            inputs = self.validate_inputs(
                original_audio=original_audio, original_text=original_text, target_text=target_text,
                edit_regions=edit_regions, device=device, model=model, seed=seed,
                fix_durations=fix_durations, temperature=temperature, speed=speed,
                target_rms=target_rms, nfe_step=nfe_step, cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef, ode_method=ode_method
            )
            
            # Load F5-TTS model
            self.load_f5tts_model(inputs["model"], inputs["device"])
            
            # Set seed for reproducibility
            self.set_seed(inputs["seed"])
            
            # Store model info for use in speech editing
            self.current_model_name = inputs["model"]
            
            # Parse edit regions and fix durations
            edit_regions_parsed = self._parse_edit_regions(inputs["edit_regions"])
            fix_durations_parsed = self._parse_fix_durations(inputs["fix_durations"])
            
            if fix_durations_parsed and len(fix_durations_parsed) != len(edit_regions_parsed):
                raise ValueError(f"Number of fix durations ({len(fix_durations_parsed)}) must match number of edit regions ({len(edit_regions_parsed)})")
            
            # Extract audio data
            if isinstance(original_audio, dict) and 'waveform' in original_audio:
                audio_tensor = original_audio['waveform']
                sample_rate = original_audio.get('sample_rate', self.f5tts_sample_rate)
            else:
                raise ValueError("Invalid audio format. Expected dictionary with 'waveform' key.")
            
            # Use F5-TTS's speech editing functionality
            edited_audio = self._perform_f5tts_edit(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                original_text=inputs["original_text"],
                target_text=inputs["target_text"],
                edit_regions=edit_regions_parsed,
                fix_durations=fix_durations_parsed,
                temperature=inputs["temperature"],
                speed=inputs["speed"],
                target_rms=inputs["target_rms"],
                nfe_step=inputs["nfe_step"],
                cfg_strength=inputs["cfg_strength"],
                sway_sampling_coef=inputs["sway_sampling_coef"],
                ode_method=inputs["ode_method"]
            )
            
            # Generate info
            total_duration = edited_audio.size(-1) / self.f5tts_sample_rate
            model_info = self.get_f5tts_model_info()
            edit_info = (f"Edited {total_duration:.1f}s audio with {len(edit_regions_parsed)} regions using F5-TTS "
                        f"{model_info.get('model_name', 'unknown')} - Original: '{inputs['original_text'][:50]}...' "
                        f"-> Target: '{inputs['target_text'][:50]}...'")
            
            # Return audio in ComfyUI format
            return (
                self.format_f5tts_audio_output(edited_audio),
                edit_info
            )
        
        return self.process_with_error_handling(_process)
    
    def _perform_f5tts_edit(self, audio_tensor: torch.Tensor, sample_rate: int,
                           original_text: str, target_text: str,
                           edit_regions: List[Tuple[float, float]],
                           fix_durations: Optional[List[float]],
                           temperature: float, speed: float, target_rms: float,
                           nfe_step: int, cfg_strength: float, sway_sampling_coef: float,
                           ode_method: str) -> torch.Tensor:
        """Perform F5-TTS speech editing"""
        
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
            model_name = getattr(self, 'current_model_name', 'F5TTS_v1_Base')
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
            
            # Create edit mask and modified audio
            edited_audio, edit_mask = self._create_edit_mask_and_audio(
                audio, edit_regions, fix_durations, target_sample_rate, hop_length
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
                    generated_wave = generated_wave * rms / target_rms
                
                print(f"Generated wave: {generated_wave.shape}")
                
                return generated_wave
                
        except ImportError as e:
            raise ImportError(f"F5-TTS modules not available for speech editing: {e}")
        except Exception as e:
            raise RuntimeError(f"F5-TTS speech editing failed: {e}")
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate inputs specific to speech editing"""
        # Call base validation
        validated = super(BaseF5TTSNode, self).validate_inputs(**inputs)
        
        # Validate required inputs
        if not validated.get("original_text", "").strip():
            raise ValueError("Original text is required and cannot be empty")
        
        if not validated.get("target_text", "").strip():
            raise ValueError("Target text is required and cannot be empty")
        
        if not validated.get("edit_regions", "").strip():
            raise ValueError("Edit regions are required and cannot be empty")
        
        # Validate edit regions format
        try:
            edit_regions = self._parse_edit_regions(validated["edit_regions"])
            if not edit_regions:
                raise ValueError("At least one edit region must be specified")
        except ValueError as e:
            raise ValueError(f"Invalid edit regions: {e}")
        
        # Validate fix durations if provided
        fix_durations_str = validated.get("fix_durations", "").strip()
        if fix_durations_str:
            try:
                fix_durations = self._parse_fix_durations(fix_durations_str)
                if fix_durations and len(fix_durations) != len(edit_regions):
                    raise ValueError(f"Number of fix durations ({len(fix_durations)}) must match number of edit regions ({len(edit_regions)})")
            except ValueError as e:
                raise ValueError(f"Invalid fix durations: {e}")
        
        return validated