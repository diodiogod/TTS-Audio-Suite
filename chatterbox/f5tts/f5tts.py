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
    "F5-JP": {"repo": "Jmica/F5TTS", "exp": "JA_8500000", "step": 8499660, "ext": "pt"},
    "F5-IT": {"repo": "alien79/F5-TTS-italian", "exp": "", "step": 159600, "ext": "safetensors"},
    "F5-TH": {"repo": "VIZINTZOR/F5-TTS-THAI", "exp": "", "step": 1000000, "ext": "pt"},
    "F5-PT-BR": {"repo": "firstpixel/F5-TTS-pt-br", "exp": "pt-br", "step": 200000, "ext": "pt"},
}

def get_f5tts_models():
    """Get list of available F5-TTS models"""
    models = list(F5TTS_MODELS.keys())
    
    # Check for local models in ComfyUI models directory
    try:
        models_dir = os.path.join(folder_paths.models_dir, "F5-TTS")
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
    
    def _load_f5tts(self):
        """Load F5-TTS model and vocoder"""
        try:
            # Try to import F5-TTS
            from f5_tts.api import F5TTS
            
            # Check if we have local model directory
            if self.ckpt_dir and os.path.exists(self.ckpt_dir):
                # Find model file and vocab file in local directory
                model_file = None
                vocab_file = None
                for file in os.listdir(self.ckpt_dir):
                    if file.endswith((".safetensors", ".pt")):
                        model_file = os.path.join(self.ckpt_dir, file)
                    elif file.endswith(".txt") and "vocab" in file.lower():
                        vocab_file = os.path.join(self.ckpt_dir, file)
                
                if model_file:
                    print(f"📁 Found local model: {model_file}")
                    print(f"📁 Found local vocab: {vocab_file}")
                    
                    # Load with explicit local files - determine correct config
                    model_config = "F5TTS_Base"  # Default config
                    if "v1" in self.model_name.lower() or "1.1" in self.model_name.lower():
                        model_config = "F5TTS_v1_Base"
                    elif "e2tts" in self.model_name.lower():
                        model_config = "E2TTS_Base"
                    # Language models use base configs
                    
                    print(f"📁 Using model config: {model_config}")
                    self.f5tts_model = F5TTS(
                        model=model_config,
                        ckpt_file=model_file,
                        vocab_file=vocab_file,
                        device=self.device
                    )
                    print(f"✅ Loaded F5-TTS completely from local files")
                    return
            
            # Determine model configuration for HuggingFace models
            if self.model_name.startswith("local:"):
                # Local model
                local_name = self.model_name[6:]  # Remove "local:" prefix
                model_path = os.path.join(folder_paths.models_dir, "F5-TTS", local_name)
                
                # Find model file
                model_file = None
                vocab_file = None
                for file in os.listdir(model_path):
                    if file.endswith((".safetensors", ".pt")):
                        model_file = os.path.join(model_path, file)
                    elif file.endswith(".txt") and "vocab" in file.lower():
                        vocab_file = os.path.join(model_path, file)
                
                if not model_file:
                    raise FileNotFoundError(f"No model file found in {model_path}")
                
                print(f"📁 Found local model: {model_file}")
                print(f"📁 Found local vocab: {vocab_file}")
                
                # Load local model - determine correct config based on folder name
                model_config = "F5TTS_Base"  # Default config
                if "v1" in local_name.lower() or "1.1" in local_name.lower():
                    model_config = "F5TTS_v1_Base"
                elif "e2tts" in local_name.lower():
                    model_config = "E2TTS_Base"
                # Language models use base configs - they don't have their own
                
                print(f"📁 Using model config: {model_config}")
                self.f5tts_model = F5TTS(
                    model=model_config,
                    ckpt_file=model_file,
                    vocab_file=vocab_file,
                    device=self.device
                )
                print(f"✅ Loaded F5-TTS completely from local files")
                
            elif self.model_name in F5TTS_MODELS:
                # Pre-configured model from HuggingFace
                model_config = F5TTS_MODELS[self.model_name]
                
                # Language models need to use base configs but download from custom repos
                if self.model_name.startswith("F5-") and self.model_name not in ["F5TTS_Base", "F5TTS_v1_Base"]:
                    # Use base config but download from language-specific repo
                    config_name = "F5TTS_Base"
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
                    
                    # Download model and vocab from custom repo  
                    # Use reduced model for F5-FR to save space (1.35GB vs 5.39GB)
                    if self.model_name == "F5-FR":
                        model_filename = "model_last_reduced.pt"
                        vocab_filename = "vocab.txt"
                    elif self.model_name == "F5-JP":
                        # Japanese model is in a subfolder with different vocab name
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_{step}.{ext}"
                        vocab_filename = f"{exp_name}/vocab_updated.txt"
                    elif self.model_name == "F5-DE":
                        # German model is in F5TTS_Base subfolder
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_{step}.{ext}"
                        vocab_filename = "vocab.txt"
                    elif self.model_name == "F5-PT-BR":
                        # Brazilian Portuguese model is in pt-br subfolder, uses smaller safetensors version
                        exp_name = model_config["exp"]
                        model_filename = f"{exp_name}/model_last.safetensors"  # Use 1.35GB version instead of 5.39GB
                        # This model doesn't have its own vocab file, use original F5-TTS vocab
                        vocab_filename = None  # Will download from original F5-TTS repo
                    else:
                        model_filename = f"model_{step}.{ext}"
                        vocab_filename = "vocab.txt"
                    
                    model_file = hf_hub_download(repo_id=repo_id, filename=model_filename)
                    
                    # Handle vocab file - some models don't have their own vocab
                    if vocab_filename is None:
                        # Use original F5-TTS vocab for models that don't have their own
                        # First check if we have F5TTS_Base locally
                        local_f5tts_base = os.path.join(folder_paths.models_dir, "F5-TTS", "F5TTS_Base", "vocab.txt")
                        if os.path.exists(local_f5tts_base):
                            vocab_file = local_f5tts_base
                            print(f"📁 Using local F5TTS_Base vocab: {vocab_file}")
                        else:
                            # Download from original F5-TTS repo
                            vocab_file = hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_Base/vocab.txt")
                    else:
                        vocab_file = hf_hub_download(repo_id=repo_id, filename=vocab_filename)
                    
                    print(f"📁 Downloaded model: {model_file}")
                    print(f"📁 Downloaded vocab: {vocab_file}")
                    
                    # Load with base config but custom files
                    self.f5tts_model = F5TTS(
                        model=config_name,
                        ckpt_file=model_file,
                        vocab_file=vocab_file,
                        device=self.device
                    )
                    
                elif self.model_name.startswith("E2-"):
                    # E2 variants use E2 config
                    config_name = "E2TTS_Base"
                    print(f"📦 Loading F5-TTS model '{self.model_name}' from HuggingFace using config '{config_name}'")
                    self.f5tts_model = F5TTS(
                        model=config_name,
                        device=self.device
                    )
                else:
                    # Standard models (F5TTS_Base, F5TTS_v1_Base, E2TTS_Base)
                    print(f"📦 Loading F5-TTS model '{self.model_name}' from HuggingFace")
                    self.f5tts_model = F5TTS(
                        model=self.model_name,
                        device=self.device
                    )
            else:
                # Default fallback to HuggingFace
                print(f"📦 Loading F5-TTS model 'F5TTS_Base' from HuggingFace (fallback)")
                self.f5tts_model = F5TTS(
                    model="F5TTS_Base",
                    device=self.device
                )
                
            print(f"✅ F5-TTS model '{self.model_name}' loaded successfully")
            
        except ImportError as e:
            raise ImportError(f"F5-TTS not available. Please install F5-TTS: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load F5-TTS model '{self.model_name}': {e}")
    
    @classmethod
    def from_local(cls, ckpt_dir: str, device: str, model_name: str = "F5TTS_Base"):
        """Load from local directory following ChatterBox pattern"""
        print(f"📦 Loading local F5-TTS model from: {ckpt_dir}")
        return cls(model_name, device, ckpt_dir)
    
    @classmethod  
    def from_pretrained(cls, device: str, model_name: str = "F5TTS_Base"):
        """Load from HuggingFace following ChatterBox pattern"""
        print(f"📦 Loading F5-TTS model '{model_name}' from HuggingFace")
        return cls(model_name, device)
    
    def generate(self, text: str, ref_audio_path: str, ref_text: str, 
                 temperature: float = 0.8, speed: float = 1.0, 
                 target_rms: float = 0.1, cross_fade_duration: float = 0.15,
                 nfe_step: int = 32, cfg_strength: float = 2.0, **kwargs) -> torch.Tensor:
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
                        remove_silence=False
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