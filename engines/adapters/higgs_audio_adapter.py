"""
Higgs Audio Engine Adapter - Engine-specific adapter for Higgs Audio 2
Provides standardized interface for Higgs Audio operations in unified engine
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
# Use absolute import to avoid relative import issues in ComfyUI
import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.models.language_mapper import get_model_for_language
from engines.higgs_audio.higgs_audio import HiggsAudioEngine, HIGGS_AUDIO_MODELS


class HiggsAudioEngineAdapter:
    """Engine-specific adapter for Higgs Audio 2."""
    
    def __init__(self, node_instance):
        """
        Initialize Higgs Audio adapter.
        
        Args:
            node_instance: HiggsAudioNode or HiggsAudioSRTNode instance
        """
        self.node = node_instance
        self.engine_type = "higgs_audio"
        self.higgs_engine = HiggsAudioEngine()
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """
        Get Higgs Audio model name for specified language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'zh')
            default_model: Default model name
            
        Returns:
            Higgs Audio model name for the language
            
        Note: Higgs Audio v2 supports multiple languages with the same model
        """
        # Higgs Audio v2 3B model supports multiple languages
        # Map all language codes to the default model for now
        supported_languages = ['en', 'zh', 'zh-cn']  # Add more as confirmed
        
        if lang_code.lower() in supported_languages:
            return "higgs-audio-v2-3B"
        else:
            print(f"⚠️ Language '{lang_code}' not specifically tested with Higgs Audio, using default model")
            return "higgs-audio-v2-3B"
    
    def load_base_model(self, model_name: str, device: str):
        """
        Load base Higgs Audio model.
        
        Args:
            model_name: Model name to load
            device: Device to load model on
        """
        # Check if the model is already loaded to avoid redundant loading
        current_model = getattr(self.node, 'current_model_name', None)
        if current_model == model_name:
            print(f"💾 Higgs Audio adapter: Model '{model_name}' already loaded - skipping base model load")
            return
        
        # Determine model paths based on model name
        if model_name in HIGGS_AUDIO_MODELS:
            model_config = HIGGS_AUDIO_MODELS[model_name]
            generation_model = model_config["generation_repo"]
            tokenizer_model = model_config["tokenizer_repo"]
        else:
            # Default fallback
            generation_model = "bosonai/higgs-audio-v2-generation-3B-base"
            tokenizer_model = "bosonai/higgs-audio-v2-tokenizer"
        
        # Initialize the Higgs Audio engine
        self.higgs_engine.initialize_engine(
            model_path=generation_model,
            tokenizer_path=tokenizer_model,
            device=device
        )
        
        # Store current model for caching
        self.node.current_model_name = model_name
        print(f"✅ Higgs Audio adapter: Loaded model '{model_name}' on {device}")
    
    def load_language_model(self, model_name: str, device: str):
        """
        Load language-specific Higgs Audio model.
        
        Args:
            model_name: Language-specific model name
            device: Device to load model on
            
        Note: Higgs Audio v2 uses the same model for all languages
        """
        # For Higgs Audio, language models are the same as base models
        self.load_base_model(model_name, device)
    
    def generate_segment_audio(self, text: str, char_audio: str, char_text: str, 
                             character: str = "narrator", **params) -> torch.Tensor:
        """
        Generate Higgs Audio audio for a text segment.
        
        Args:
            text: Text to generate audio for
            char_audio: Reference audio file path or audio dict
            char_text: Reference text
            character: Character name for caching
            **params: Additional Higgs Audio parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract Higgs Audio specific parameters
        voice_preset = params.get("voice_preset", "voice_clone")
        system_prompt = params.get("system_prompt", "Generate audio following instruction.")
        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 0.95)
        top_k = params.get("top_k", 50)
        max_new_tokens = params.get("max_new_tokens", 2048)
        seed = params.get("seed", -1)
        enable_cache = params.get("enable_audio_cache", True)
        audio_priority = params.get("audio_priority", "auto")
        
        # Prepare reference audio
        reference_audio = None
        if char_audio:
            if isinstance(char_audio, str) and os.path.exists(char_audio):
                # Load audio file
                try:
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(char_audio)
                    if waveform.dim() == 2 and waveform.size(0) > 1:
                        # Convert to mono if stereo
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    reference_audio = {"waveform": waveform.float(), "sample_rate": sample_rate}
                except Exception as e:
                    print(f"⚠️ Failed to load reference audio {char_audio}: {e}")
            elif isinstance(char_audio, dict):
                # Already in ComfyUI format
                reference_audio = char_audio
        
        # Generate audio using Higgs Audio engine
        try:
            audio_result, generation_info = self.higgs_engine.generate(
                text=text,
                voice_preset=voice_preset,
                reference_audio=reference_audio,
                reference_text=char_text or "",
                audio_priority=audio_priority,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                enable_chunking=False,  # Single segment, no chunking needed
                enable_cache=enable_cache,
                character=character,
                seed=seed
            )
            
            # Return the waveform tensor
            return audio_result["waveform"]
            
        except Exception as e:
            print(f"❌ Error generating Higgs Audio segment: {e}")
            raise e
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Higgs Audio models.
        
        Returns:
            List of model names
        """
        return self.higgs_engine.get_available_models()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a Higgs Audio model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dict or None
        """
        return self.higgs_engine.downloader.get_model_info(model_name)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'higgs_engine'):
            self.higgs_engine.cleanup()
        
        # Clear current model reference
        if hasattr(self.node, 'current_model_name'):
            self.node.current_model_name = None
    
    def supports_streaming(self) -> bool:
        """
        Check if Higgs Audio supports streaming.
        
        Returns:
            False for now - streaming not implemented yet
        """
        return False
    
    def get_sample_rate(self) -> int:
        """
        Get the sample rate used by Higgs Audio.
        
        Returns:
            Sample rate (24000 Hz for Higgs Audio v2)
        """
        return 24000  # Higgs Audio v2 uses 24kHz sample rate
    
    def supports_voice_cloning(self) -> bool:
        """
        Check if Higgs Audio supports voice cloning.
        
        Returns:
            True - Higgs Audio v2 supports voice cloning
        """
        return True
    
    def get_voice_presets(self) -> Dict[str, str]:
        """
        Get available voice presets.
        
        Returns:
            Dictionary of voice preset names and descriptions
        """
        presets, _ = self.higgs_engine.load_voice_presets()
        return presets
    
    def validate_parameters(self, **params) -> Dict[str, Any]:
        """
        Validate and normalize Higgs Audio parameters.
        
        Args:
            **params: Parameters to validate
            
        Returns:
            Validated parameters dict
        """
        validated = {}
        
        # Temperature validation
        temperature = params.get("temperature", 1.0)
        validated["temperature"] = max(0.0, min(2.0, float(temperature)))
        
        # Top-p validation
        top_p = params.get("top_p", 0.95)
        validated["top_p"] = max(0.1, min(1.0, float(top_p)))
        
        # Top-k validation
        top_k = params.get("top_k", 50)
        validated["top_k"] = max(-1, min(100, int(top_k)))
        
        # Max tokens validation
        max_tokens = params.get("max_new_tokens", 2048)
        validated["max_new_tokens"] = max(128, min(4096, int(max_tokens)))
        
        # Voice preset validation
        voice_preset = params.get("voice_preset", "voice_clone")
        available_presets = list(self.get_voice_presets().keys())
        if voice_preset not in available_presets:
            print(f"⚠️ Unknown voice preset '{voice_preset}', using 'voice_clone'")
            validated["voice_preset"] = "voice_clone"
        else:
            validated["voice_preset"] = voice_preset
        
        # Audio priority validation
        audio_priority = params.get("audio_priority", "auto")
        valid_priorities = ["auto", "preset_dropdown", "reference_input", "force_preset"]
        if audio_priority not in valid_priorities:
            print(f"⚠️ Invalid audio priority '{audio_priority}', using 'auto'")
            validated["audio_priority"] = "auto"
        else:
            validated["audio_priority"] = audio_priority
        
        # System prompt
        validated["system_prompt"] = params.get("system_prompt", "Generate audio following instruction.")
        
        # Seed validation
        seed = params.get("seed", -1)
        validated["seed"] = int(seed) if seed >= 0 else -1
        
        # Cache settings
        validated["enable_audio_cache"] = params.get("enable_audio_cache", True)
        
        return validated