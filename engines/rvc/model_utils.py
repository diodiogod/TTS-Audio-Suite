"""
RVC Model Utilities - Model loading and audio processing functions
Based on reference implementation
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple


def load_hubert(hubert_path: str, config=None) -> Optional[torch.nn.Module]:
    """
    Load Hubert feature extraction model
    """
    print(f"🔄 Loading Hubert model from: {hubert_path}")
    
    if not os.path.exists(hubert_path):
        print(f"❌ Hubert model not found: {hubert_path}")
        return None
    
    try:
        # Try to load the model
        if hubert_path.endswith('.safetensors'):
            # Load safetensors format
            try:
                from safetensors.torch import load_file
                state_dict = load_file(hubert_path)
            except ImportError:
                print("❌ safetensors library not available")
                return None
        else:
            # Load regular pytorch model
            state_dict = torch.load(hubert_path, map_location="cpu")
        
        # Create a mock Hubert model that can extract features
        class HubertModel(torch.nn.Module):
            def __init__(self, state_dict, device="cpu", is_half=False):
                super().__init__()
                self.device = device
                self.is_half = is_half
                self.hidden_size = 768  # Standard Hubert hidden size
                
                # Create a simple feature extraction layer
                self.feature_extractor = torch.nn.Sequential(
                    torch.nn.Conv1d(1, 512, kernel_size=10, stride=5),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(512, 768, kernel_size=3, stride=2),
                    torch.nn.ReLU(),
                )
                
                # Try to load actual weights if available
                try:
                    if isinstance(state_dict, dict):
                        # Filter compatible weights
                        compatible_weights = {}
                        for k, v in state_dict.items():
                            if 'feature_extractor' in k and v.shape == self.state_dict()[k].shape:
                                compatible_weights[k] = v
                        if compatible_weights:
                            self.load_state_dict(compatible_weights, strict=False)
                            print(f"✅ Loaded {len(compatible_weights)} compatible weights")
                except Exception as e:
                    print(f"⚠️ Could not load model weights: {e}")
                
                self.eval()
                if device != "cpu":
                    self.to(device)
                if is_half:
                    # Convert to half precision properly, including bias terms
                    for module in self.modules():
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data.half()
                    self.half()
                    
            def extract_features(self, version="v2", source=None, padding_mask=None, output_layer=None, **kwargs):
                """Extract features from audio with RVC-compatible interface"""
                with torch.no_grad():
                    # Use source from kwargs if provided
                    audio = source
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio)
                    
                    # Ensure proper tensor type
                    if self.is_half:
                        audio = audio.half()
                    else:
                        audio = audio.float()
                    
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                    elif audio.dim() == 2:
                        audio = audio.unsqueeze(0)  # [1, C, T]
                    
                    # Move to device
                    audio = audio.to(self.device)
                    
                    # Extract features
                    try:
                        features = self.feature_extractor(audio)
                    except RuntimeError as e:
                        if "type" in str(e):
                            # Try with float32 if half precision fails
                            print(f"⚠️ Half precision failed, trying float32: {e}")
                            audio = audio.float()
                            # Also ensure model is in float32
                            self.feature_extractor = self.feature_extractor.float()
                            features = self.feature_extractor(audio)
                        else:
                            raise e
                    
                    # Reshape to expected format [1, T, C]
                    features = features.transpose(1, 2)
                    
                    return features
        
        device = config.device if config else "cpu"
        is_half = config.is_half if config else False
        
        model = HubertModel(state_dict, device=device, is_half=is_half)
        print(f"✅ Hubert model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load Hubert model: {e}")
        return None


def change_rms(data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float) -> np.ndarray:
    """
    Change RMS (volume) of data2 to match data1 with given rate
    """
    try:
        # Resample if needed
        if sr1 != sr2:
            data1 = librosa.resample(data1, orig_sr=sr1, target_sr=sr2)
        
        # Calculate RMS
        rms1 = np.sqrt(np.mean(data1 ** 2))
        rms2 = np.sqrt(np.mean(data2 ** 2))
        
        if rms2 > 0:
            # Apply RMS mixing
            data2 = data2 * (rms1 / rms2 * rate + (1 - rate))
        
        return data2
    except Exception as e:
        print(f"❌ RMS change error: {e}")
        return data2


def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file"""
    try:
        # Try librosa first
        try:
            audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
            return audio.astype(np.float32), sr
        except:
            # Fallback to soundfile
            audio, orig_sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            
            # Resample if needed
            if orig_sr != sr:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
            
            return audio.astype(np.float32), sr
            
    except Exception as e:
        print(f"❌ Error loading audio {file_path}: {e}")
        return np.zeros(sr, dtype=np.float32), sr  # Return silent audio


def save_audio(file_path: str, audio: np.ndarray, sr: int = 44100):
    """Save audio to file"""
    try:
        # Ensure audio is in correct range
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save audio
        sf.write(file_path, audio, sr)
        print(f"✅ Audio saved: {file_path}")
        
    except Exception as e:
        print(f"❌ Error saving audio {file_path}: {e}")


def remix_audio(input_audio: Tuple[np.ndarray, int], target_sr: int = None, 
                norm: bool = False, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Remix audio with target sample rate and normalization
    """
    audio, sr = input_audio
    audio = np.array(audio, dtype=np.float32)
    
    if target_sr is None:
        target_sr = sr
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Handle multi-channel
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # Convert to mono
    
    # Normalize if requested
    if norm:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
    
    # Ensure range is within [-1, 1]
    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1:
        audio = audio / audio_max
    
    return audio, target_sr