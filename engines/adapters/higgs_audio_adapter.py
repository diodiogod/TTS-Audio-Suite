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
from engines.higgs_audio.higgs_audio import HiggsAudioEngine
from engines.higgs_audio.higgs_audio_downloader import HIGGS_AUDIO_MODELS


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
    
    def load_base_model(self, model_name: str, device: str, enable_cuda_graphs: bool = True):
        """
        Load base Higgs Audio model.
        
        Args:
            model_name: Model name to load
            device: Device to load model on
            enable_cuda_graphs: Whether to enable CUDA Graph optimization
        """
        # Check if the model is already loaded AND engine is initialized
        # Also check for corruption flags that require recreation
        current_model = getattr(self.node, 'current_model_name', None)
        cuda_graph_corrupted = getattr(self.higgs_engine, '_cuda_graph_corrupted', False)
        needs_recreation = getattr(self.higgs_engine, '_needs_recreation', False)
        
        if (current_model == model_name and 
            self.higgs_engine.engine is not None and 
            not cuda_graph_corrupted and 
            not needs_recreation):
            print(f"💾 Higgs Audio adapter: Model '{model_name}' already loaded - skipping base model load")
            return
        
        if cuda_graph_corrupted or needs_recreation:
            print(f"🔥 Higgs Audio adapter: Engine corruption detected - forcing complete recreation")
            # Clear the corruption flags
            self.higgs_engine._cuda_graph_corrupted = False
            self.higgs_engine._needs_recreation = False
            # Force new engine creation by clearing current engine
            self.higgs_engine.engine = None
        
        # Determine model paths based on model name
        if model_name in HIGGS_AUDIO_MODELS:
            model_config = HIGGS_AUDIO_MODELS[model_name]
            generation_model = model_config["generation_repo"]
            tokenizer_model = model_config["tokenizer_repo"]
        else:
            # Default fallback
            generation_model = "bosonai/higgs-audio-v2-generation-3B-base"
            tokenizer_model = "bosonai/higgs-audio-v2-tokenizer"
        
        # Initialize the Higgs Audio engine with CUDA Graph setting
        self.higgs_engine.initialize_engine(
            model_path=generation_model,
            tokenizer_path=tokenizer_model,
            device=device,
            enable_cuda_graphs=enable_cuda_graphs
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
        # Ensure engine is initialized with current model/device
        model = params.get("model", "higgs-audio-v2-3B")
        device = params.get("device", "auto")
        enable_cuda_graphs = params.get("enable_cuda_graphs", True)  # Get CUDA Graph setting
        self.load_base_model(model, device, enable_cuda_graphs)
        
        # Extract Higgs Audio specific parameters  
        system_prompt = params.get("system_prompt", "Generate audio following instruction.")
        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 0.95)
        top_k = params.get("top_k", 50)
        max_new_tokens = params.get("max_new_tokens", 2048)
        force_audio_gen = params.get("force_audio_gen", False)
        ras_win_len = params.get("ras_win_len", 7)  # Default from boson_multimodal
        ras_max_num_repeat = params.get("ras_max_num_repeat", 2)  # Default from boson_multimodal
        enable_cuda_graphs = params.get("enable_cuda_graphs", True)  # CUDA Graph toggle
        seed = params.get("seed", -1)
        enable_cache = params.get("enable_audio_cache", True)
        model_name = params.get("model", "higgs-audio-v2-3B")
        device = params.get("device", "auto")
        
        # Native multi-speaker mode parameters
        multi_speaker_mode = params.get("multi_speaker_mode", "Custom Character Switching")
        second_narrator_audio = params.get("second_narrator_audio")
        second_narrator_text = params.get("second_narrator_text", "")
        
        # Chunking parameters
        max_chars_per_chunk = params.get("max_chars_per_chunk", 400)
        silence_between_chunks_ms = params.get("silence_between_chunks_ms", 100)
        enable_chunking = len(text) > max_chars_per_chunk  # Auto-enable chunking for long text
        
        # Convert chars to tokens for engine (Higgs Audio uses ~3.5 chars per token)
        max_tokens_per_chunk = int(max_chars_per_chunk / 3.5)
        
        # Initialize engine if not already done
        if not self.higgs_engine.engine:
            print(f"🚀 Initializing Higgs Audio engine with model: {model_name}")
            self.load_base_model(model_name, device)
        
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
                    
                    # Move to same device as the model to prevent device mismatch
                    target_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                    waveform = waveform.float().to(target_device)
                    
                    reference_audio = {"waveform": waveform, "sample_rate": sample_rate}
                    print(f"  📁 Loaded reference audio on {target_device}")
                except Exception as e:
                    print(f"⚠️ Failed to load reference audio {char_audio}: {e}")
            elif isinstance(char_audio, dict):
                # Already in ComfyUI format - ensure tensors are on correct device
                reference_audio = {}
                target_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                
                for key, value in char_audio.items():
                    if torch.is_tensor(value):
                        reference_audio[key] = value.to(target_device)
                        print(f"  📦 Moved ComfyUI {key} tensor to {target_device}")
                    else:
                        reference_audio[key] = value
        
        # Generate audio using Higgs Audio engine with stateless methods
        try:
            if multi_speaker_mode in ["Native Multi-Speaker (System Context)", "Native Multi-Speaker (Conversation)"]:
                # Use stateless native multi-speaker mode with multiple reference audios
                if hasattr(self.higgs_engine, 'generate_native_multispeaker_stateless'):
                    audio_result, generation_info = self.higgs_engine.generate_native_multispeaker_stateless(
                        text=text,
                        primary_reference_audio=reference_audio,
                        primary_reference_text=char_text or "",
                        secondary_reference_audio=second_narrator_audio,
                        secondary_reference_text=second_narrator_text,
                        use_system_context=(multi_speaker_mode == "Native Multi-Speaker (System Context)"),
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        force_audio_gen=force_audio_gen,
                        ras_win_len=ras_win_len,
                        ras_max_num_repeat=ras_max_num_repeat,
                        enable_cache=enable_cache,
                        character=character,
                        seed=seed
                    )
                else:
                    # Fallback to standard method if wrapper not available
                    audio_result, generation_info = self.higgs_engine.generate_native_multispeaker(
                        text=text,
                        primary_reference_audio=reference_audio,
                        primary_reference_text=char_text or "",
                        secondary_reference_audio=second_narrator_audio,
                        secondary_reference_text=second_narrator_text,
                        use_system_context=(multi_speaker_mode == "Native Multi-Speaker (System Context)"),
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        force_audio_gen=force_audio_gen,
                        ras_win_len=ras_win_len,
                        ras_max_num_repeat=ras_max_num_repeat,
                        enable_cache=enable_cache,
                        character=character,
                        seed=seed
                    )
            else:
                # Use stateless custom character switching mode (existing behavior)
                if hasattr(self.higgs_engine, 'generate_stateless'):
                    audio_result, generation_info = self.higgs_engine.generate_stateless(
                        text=text,
                        reference_audio=reference_audio,
                        reference_text=char_text or "",
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        force_audio_gen=force_audio_gen,
                        ras_win_len=ras_win_len,
                        ras_max_num_repeat=ras_max_num_repeat,
                        enable_chunking=enable_chunking,
                        max_tokens_per_chunk=max_tokens_per_chunk,  # Now properly converted from chars to tokens
                        silence_between_chunks_ms=silence_between_chunks_ms,
                        enable_cache=enable_cache,
                        character=character,
                        seed=seed
                    )
                else:
                    # Fallback to standard method if wrapper not available
                    audio_result, generation_info = self.higgs_engine.generate(
                        text=text,
                        reference_audio=reference_audio,
                        reference_text=char_text or "",
                        system_prompt=system_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        force_audio_gen=force_audio_gen,
                        ras_win_len=ras_win_len,
                        ras_max_num_repeat=ras_max_num_repeat,
                        enable_chunking=enable_chunking,
                        max_tokens_per_chunk=max_tokens_per_chunk,  # Now properly converted from chars to tokens
                        silence_between_chunks_ms=silence_between_chunks_ms,
                        enable_cache=enable_cache,
                        character=character,
                        seed=seed
                    )
            
            # Return the waveform tensor with correct dimensions for ComfyUI
            waveform = audio_result["waveform"]
            
            # Ensure correct dimensions: ComfyUI expects [channels, samples]
            if waveform.dim() == 3:
                # Handle [batch, channels, samples] -> [channels, samples]
                if waveform.size(0) == 1:  # batch size of 1
                    waveform = waveform.squeeze(0)  # Remove batch dimension
                else:
                    waveform = waveform[0]  # Take first item from batch
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            
            return waveform
            
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
        validated["max_new_tokens"] = max(1, min(4096, int(max_tokens)))
        
        # Force audio gen validation
        force_audio_gen = params.get("force_audio_gen", False)
        validated["force_audio_gen"] = bool(force_audio_gen)
        
        # RAS parameters validation
        ras_win_len = params.get("ras_win_len", 7)
        if ras_win_len is not None and ras_win_len > 0:
            validated["ras_win_len"] = max(1, min(20, int(ras_win_len)))
        else:
            validated["ras_win_len"] = None  # Disable RAS
        
        ras_max_num_repeat = params.get("ras_max_num_repeat", 2)
        validated["ras_max_num_repeat"] = max(1, min(5, int(ras_max_num_repeat)))
        
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