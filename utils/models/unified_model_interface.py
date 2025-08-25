"""
Unified Model Interface for TTS Audio Suite

This module provides a standardized interface for all TTS engines to use
ComfyUI's model management system. All engines (ChatterBox, F5-TTS, Higgs Audio, 
RVC, etc.) should use this interface instead of direct caching.
"""

import torch
from typing import Any, Dict, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import Path

from utils.models.comfyui_model_wrapper import tts_model_manager, ModelInfo


@dataclass
class UnifiedModelConfig:
    """Configuration for unified model loading"""
    engine_name: str        # "chatterbox", "f5tts", "higgs_audio", "rvc"
    model_type: str         # "tts", "vc", "tokenizer", "hubert", "separator"
    model_name: str         # Model identifier
    device: str             # Target device
    language: Optional[str] = "English"  # For multilingual models
    model_path: Optional[str] = None     # Local path if available
    repo_id: Optional[str] = None        # HuggingFace repo ID
    additional_params: Optional[Dict[str, Any]] = None  # Engine-specific params


class UnifiedModelInterface:
    """
    Unified interface for all TTS engine model loading.
    
    This replaces all engine-specific caching systems with a single
    ComfyUI-integrated approach.
    """
    
    def __init__(self):
        """Initialize the unified interface"""
        self._model_factories: Dict[str, Callable] = {}
        
    def register_model_factory(self, 
                             engine_name: str, 
                             model_type: str, 
                             factory_func: Callable) -> None:
        """
        Register a model factory function for an engine.
        
        Args:
            engine_name: Name of the engine ("chatterbox", "f5tts", etc.)
            model_type: Type of model ("tts", "vc", "tokenizer", etc.)
            factory_func: Function that creates the model
        """
        key = f"{engine_name}_{model_type}"
        self._model_factories[key] = factory_func
        print(f"📝 Registered model factory: {key}")
    
    def load_model(self, config: UnifiedModelConfig, force_reload: bool = False) -> Any:
        """
        Load a model using the unified interface.
        
        Args:
            config: Model configuration
            force_reload: Force reload even if cached
            
        Returns:
            The loaded model (unwrapped from ComfyUI wrapper)
            
        Raises:
            ValueError: If no factory is registered for the engine/model type
            RuntimeError: If model loading fails
        """
        # Generate unique cache key
        cache_key = self._generate_cache_key(config)
        
        # Check if force reload is needed
        if force_reload:
            tts_model_manager.remove_model(cache_key)
        
        # Get existing model if cached
        wrapper = tts_model_manager.get_model(cache_key)
        if wrapper is not None:
            # Ensure model is on correct device
            if wrapper.current_device != config.device and config.device != "auto":
                wrapper.model_load(config.device)
            return wrapper.model
        
        # Find appropriate factory
        factory_key = f"{config.engine_name}_{config.model_type}"
        if factory_key not in self._model_factories:
            raise ValueError(f"No model factory registered for {factory_key}")
        
        factory_func = self._model_factories[factory_key]
        
        # Prepare factory arguments (device will be added by model manager)
        factory_kwargs = {
            "model_name": config.model_name,
            "language": config.language,
            "model_path": config.model_path,
            "repo_id": config.repo_id,
        }
        
        # Add engine-specific parameters
        if config.additional_params:
            factory_kwargs.update(config.additional_params)
        
        # Use ComfyUI model manager to load
        wrapper = tts_model_manager.load_model(
            model_factory_func=factory_func,
            model_key=cache_key,
            model_type=config.model_type,
            engine=config.engine_name,
            device=config.device,
            force_reload=force_reload,
            **factory_kwargs
        )
        
        return wrapper.model
    
    def unload_model(self, config: UnifiedModelConfig) -> bool:
        """
        Unload a specific model.
        
        Args:
            config: Model configuration to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        cache_key = self._generate_cache_key(config)
        return tts_model_manager.remove_model(cache_key)
    
    def clear_engine_models(self, engine_name: str) -> None:
        """Clear all models for a specific engine"""
        tts_model_manager.clear_cache(engine=engine_name)
    
    def clear_model_type(self, model_type: str) -> None:
        """Clear all models of a specific type"""
        tts_model_manager.clear_cache(model_type=model_type)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        return tts_model_manager.get_stats()
    
    def _generate_cache_key(self, config: UnifiedModelConfig) -> str:
        """Generate unique cache key for model configuration"""
        components = [
            config.engine_name,
            config.model_type,
            config.model_name,
            config.device,
            config.language or "default",
        ]
        
        # Add path/repo info if available
        if config.model_path:
            components.append(f"path:{Path(config.model_path).name}")
        elif config.repo_id:
            components.append(f"repo:{config.repo_id}")
        
        cache_key = "_".join(components)
        print(f"🔑 Generated cache key: {cache_key}")
        return cache_key


# Global unified interface instance
unified_model_interface = UnifiedModelInterface()


# Convenience functions for common model operations
def load_tts_model(engine_name: str, 
                   model_name: str, 
                   device: str, 
                   language: str = "English",
                   model_path: Optional[str] = None,
                   repo_id: Optional[str] = None,
                   force_reload: bool = False,
                   **kwargs) -> Any:
    """
    Convenience function to load TTS models.
    
    Args:
        engine_name: Engine name ("chatterbox", "f5tts", etc.)
        model_name: Model identifier
        device: Target device
        language: Model language
        model_path: Local path if available
        repo_id: HuggingFace repo ID if available
        force_reload: Force reload even if cached
        **kwargs: Additional engine-specific parameters
        
    Returns:
        Loaded TTS model
    """
    config = UnifiedModelConfig(
        engine_name=engine_name,
        model_type="tts",
        model_name=model_name,
        device=device,
        language=language,
        model_path=model_path,
        repo_id=repo_id,
        additional_params=kwargs
    )
    return unified_model_interface.load_model(config, force_reload)


def load_vc_model(engine_name: str,
                  model_name: str,
                  device: str,
                  language: str = "English",
                  model_path: Optional[str] = None,
                  repo_id: Optional[str] = None,
                  force_reload: bool = False,
                  **kwargs) -> Any:
    """
    Convenience function to load Voice Conversion models.
    """
    config = UnifiedModelConfig(
        engine_name=engine_name,
        model_type="vc",
        model_name=model_name,
        device=device,
        language=language,
        model_path=model_path,
        repo_id=repo_id,
        additional_params=kwargs
    )
    return unified_model_interface.load_model(config, force_reload)


def load_auxiliary_model(engine_name: str,
                        model_type: str,  # "tokenizer", "hubert", "separator", etc.
                        model_name: str,
                        device: str,
                        model_path: Optional[str] = None,
                        repo_id: Optional[str] = None,
                        force_reload: bool = False,
                        **kwargs) -> Any:
    """
    Convenience function to load auxiliary models (tokenizers, HuBERT, etc.).
    """
    config = UnifiedModelConfig(
        engine_name=engine_name,
        model_type=model_type,
        model_name=model_name,
        device=device,
        model_path=model_path,
        repo_id=repo_id,
        additional_params=kwargs
    )
    return unified_model_interface.load_model(config, force_reload)


# Factory registration helpers
def register_chatterbox_factory():
    """Register ChatterBox model factories"""
    def chatterbox_tts_factory(**kwargs):
        # Import here to avoid circular imports
        from engines.chatterbox.tts import ChatterboxTTS
        
        device = kwargs.get("device", "auto")
        language = kwargs.get("language", "English")
        model_path = kwargs.get("model_path")
        
        if model_path:
            return ChatterboxTTS.from_local(model_path, device)
        else:
            return ChatterboxTTS.from_pretrained(device, language)
    
    def chatterbox_vc_factory(**kwargs):
        from engines.chatterbox.vc import ChatterboxVC
        
        device = kwargs.get("device", "auto")
        language = kwargs.get("language", "English")
        model_path = kwargs.get("model_path")
        
        if model_path:
            return ChatterboxVC.from_local(model_path, device)
        else:
            return ChatterboxVC.from_pretrained(device, language)
    
    unified_model_interface.register_model_factory("chatterbox", "tts", chatterbox_tts_factory)
    unified_model_interface.register_model_factory("chatterbox", "vc", chatterbox_vc_factory)


def register_f5tts_factory():
    """Register F5-TTS model factories"""
    def f5tts_factory(**kwargs):
        from engines.f5tts.f5tts import ChatterBoxF5TTS
        
        device = kwargs.get("device", "auto")
        model_name = kwargs.get("model_name", "F5TTS_Base")
        
        return ChatterBoxF5TTS.from_pretrained(device, model_name)
    
    unified_model_interface.register_model_factory("f5tts", "tts", f5tts_factory)


def register_higgs_audio_factory():
    """Register Higgs Audio model factories"""
    def higgs_audio_factory(**kwargs):
        from engines.higgs_audio.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        model_name = kwargs.get("model_name")
        device = kwargs.get("device", "cuda")
        model_path = kwargs.get("model_path", model_name)
        tokenizer_path = kwargs.get("tokenizer_path")
        
        return HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=tokenizer_path or model_path,
            device=device,
            kv_cache_lengths=[1024, 2048, 4096]  # Memory-efficient cache sizes
        )
    
    unified_model_interface.register_model_factory("higgs_audio", "tts", higgs_audio_factory)


def register_rvc_factory():
    """Register RVC model factories"""  
    def rvc_factory(**kwargs):
        from engines.rvc.rvc_engine import RVCEngine
        return RVCEngine()
    
    def hubert_factory(**kwargs):
        from engines.rvc.impl.lib.model_utils import load_hubert
        model_path = kwargs.get("model_path")
        config = kwargs.get("config", {})
        return load_hubert(model_path, config)
    
    def mdx23c_separator_factory(**kwargs):
        from engines.rvc.impl.lib.mdx23c import MDX23C
        model_path = kwargs.get("model_path")
        device = kwargs.get("device", "cuda")
        return MDX23C(model_path, device)
    
    def demucs_separator_factory(**kwargs):
        from engines.rvc.impl.lib.uvr5_pack.demucs.pretrained import get_model
        model_name = kwargs.get("model_name", "htdemucs")
        device = kwargs.get("device", "cuda")
        return get_model(model_name).to(device)
    
    unified_model_interface.register_model_factory("rvc", "vc", rvc_factory)
    unified_model_interface.register_model_factory("rvc", "hubert", hubert_factory)
    unified_model_interface.register_model_factory("rvc", "separator_mdx", mdx23c_separator_factory)
    unified_model_interface.register_model_factory("rvc", "separator_demucs", demucs_separator_factory)


def initialize_all_factories():
    """Initialize all model factories"""
    register_chatterbox_factory()
    register_f5tts_factory() 
    register_higgs_audio_factory()
    register_rvc_factory()
    print("✅ All model factories registered with unified interface")


# Auto-initialize on import
initialize_all_factories()