"""
ComfyUI Custom Nodes for ChatterboxTTS - Voice Edition
Enhanced with unique naming to avoid conflicts
"""

import torch
import torchaudio
import numpy as np
import folder_paths
import os
import tempfile

try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.vc import ChatterboxVC
    CHATTERBOX_AVAILABLE = True
except ImportError as e:
    print(f"❌ ChatterBox import failed: {e}")
    print("💡 Missing dependency. Install with: pip install [missing_package]")
    CHATTERBOX_AVAILABLE = False
    
    # Create dummy classes so ComfyUI doesn't crash
    class ChatterboxTTS:
        @classmethod
        def from_pretrained(cls, device):
            raise ImportError("ChatterboxTTS not available - install missing dependencies")
    
    class ChatterboxVC:
        @classmethod 
        def from_pretrained(cls, device):
            raise ImportError("ChatterboxVC not available - install missing dependencies")

class ChatterboxTTSNode:
    """
    Text-to-Speech node using ChatterboxTTS - Voice Edition
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello world! This is ChatterboxTTS Voice Edition in ComfyUI."
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "exaggeration": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.25, 
                    "max": 2.0, 
                    "step": 0.05
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.05, 
                    "max": 5.0, 
                    "step": 0.05
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "audio_prompt_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox Voice"  # UPDATED: Unique category

    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            print(f"Loading ChatterboxTTS model on {device}...")
            
            # Try local models folder first (ComfyUI standard)
            local_model_path = os.path.join(folder_paths.models_dir, "TTS", "chatterbox")
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                print(f"📁 Loading from local path: {local_model_path}")
                self.model = ChatterboxTTS.from_local(local_model_path, device)
            else:
                print("🌐 Loading from Hugging Face (requires authentication)...")
                self.model = ChatterboxTTS.from_pretrained(device)
            
            self.device = device
            print("✅ ChatterboxTTS model loaded!")

    def generate_speech(self, text, device, exaggeration, temperature, cfg_weight, seed, reference_audio=None, audio_prompt_path=""):
        self.load_model(device)
        
        # Set seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        # Handle reference audio input
        audio_prompt = None
        if reference_audio is not None:
            # Save reference audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Handle ComfyUI audio format (may have batch dimension)
                waveform = reference_audio["waveform"]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)  # Remove batch dimension if present
                torchaudio.save(tmp_file.name, waveform, reference_audio["sample_rate"])
                audio_prompt = tmp_file.name
        elif audio_prompt_path and os.path.exists(audio_prompt_path):
            audio_prompt = audio_prompt_path

        # Generate speech
        wav = self.model.generate(
            text,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )

        # Clean up temporary file
        if reference_audio is not None and audio_prompt:
            try:
                os.unlink(audio_prompt)
            except:
                pass

        # Return audio in ComfyUI format
        return ({
            "waveform": wav.unsqueeze(0),  # Add batch dimension
            "sample_rate": self.model.sr
        },)


class ChatterboxVCNode:
    """
    Voice Conversion node using ChatterboxVC - Voice Edition
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("converted_audio",)
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox Voice"  # UPDATED: Unique category

    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self, device):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            print(f"Loading ChatterboxVC model on {device}...")
            
            # Try local models folder first (ComfyUI standard)  
            local_model_path = os.path.join(folder_paths.models_dir, "TTS", "chatterbox")
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                print(f"📁 Loading from local path: {local_model_path}")
                self.model = ChatterboxVC.from_local(local_model_path, device)
            else:
                print("🌐 Loading from Hugging Face (requires authentication)...")
                self.model = ChatterboxVC.from_pretrained(device)
            
            self.device = device
            print("✅ ChatterboxVC model loaded!")

    def convert_voice(self, source_audio, target_audio, device):
        self.load_model(device)

        # Save audio to temporary files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as source_tmp:
            source_waveform = source_audio["waveform"]
            if source_waveform.dim() == 3:
                source_waveform = source_waveform.squeeze(0)  # Remove batch dimension if present
            torchaudio.save(source_tmp.name, source_waveform, source_audio["sample_rate"])
            source_path = source_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as target_tmp:
            target_waveform = target_audio["waveform"]
            if target_waveform.dim() == 3:
                target_waveform = target_waveform.squeeze(0)  # Remove batch dimension if present
            torchaudio.save(target_tmp.name, target_waveform, target_audio["sample_rate"])
            target_path = target_tmp.name

        try:
            # Perform voice conversion
            wav = self.model.generate(
                source_path,
                target_voice_path=target_path
            )

            # Clean up temporary files
            os.unlink(source_path)
            os.unlink(target_path)

            # Return audio in ComfyUI format
            return ({
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": self.model.sr
            },)

        except Exception as e:
            # Clean up on error
            try:
                os.unlink(source_path)
                os.unlink(target_path)
            except:
                pass
            raise e


# Node mappings for ComfyUI - UPDATED: Unique names to avoid conflicts
NODE_CLASS_MAPPINGS = {
    "ChatterBoxVoiceTTS": ChatterboxTTSNode,
    "ChatterBoxVoiceVC": ChatterboxVCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxVoiceTTS": "🎤 ChatterBox Voice TTS",
    "ChatterBoxVoiceVC": "🔄 ChatterBox Voice Conversion", 
}