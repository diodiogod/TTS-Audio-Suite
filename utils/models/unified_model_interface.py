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
        # print(f"🔑 Generated cache key: {cache_key}")  # Debug only when needed
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
        from utils.models.smart_loader import smart_model_loader
        
        device = kwargs.get("device", "auto")
        model_name = kwargs.get("model_name", "F5TTS_Base")
        
        # Use Smart Loader for consistency with the base node approach
        def f5tts_load_callback(device: str, model: str) -> Any:
            """Factory callback for F5-TTS model loading"""
            # Try local first, then HuggingFace
            try:
                # Check for local models
                import os
                import folder_paths
                search_paths = [
                    os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", model),
                    os.path.join(folder_paths.models_dir, "F5-TTS", model),  # Legacy
                    os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS", model)  # Legacy
                ]
                
                local_path = None
                for path in search_paths:
                    if os.path.exists(path):
                        local_path = path
                        break
                
                if local_path:
                    print(f"📁 Using local F5-TTS model: {local_path}")
                    return ChatterBoxF5TTS.from_local(local_path, device, model)
                else:
                    print(f"📥 Loading F5-TTS model '{model}' from HuggingFace")
                    return ChatterBoxF5TTS.from_pretrained(device, model)
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load F5-TTS model '{model}': {e}")
        
        model, _ = smart_model_loader.load_model_if_needed(
            engine_type="f5tts",
            model_name=model_name,
            current_model=None,  # Factory always loads fresh
            device=device,
            load_callback=f5tts_load_callback,
            force_reload=False
        )
        
        return model
    
    unified_model_interface.register_model_factory("f5tts", "tts", f5tts_factory)


def register_higgs_audio_factory():
    """Register Higgs Audio model factories with stateless wrapper for ComfyUI integration"""
    def higgs_audio_factory(**kwargs):
        from engines.higgs_audio.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        from engines.higgs_audio.higgs_audio import HiggsAudioEngine
        from engines.higgs_audio.stateless_wrapper import create_stateless_higgs_wrapper
        
        model_name = kwargs.get("model_name")
        device = kwargs.get("device", "cuda")
        model_path = kwargs.get("model_path", model_name)
        tokenizer_path = kwargs.get("tokenizer_path")
        enable_cuda_graphs = kwargs.get("enable_cuda_graphs", True)
        
        # Check if we need to force recreate due to CUDA Graph corruption
        force_recreate = kwargs.get("_force_recreate", False)
        if force_recreate:
            print(f"🔥 Force recreating Higgs Audio engine due to CUDA Graph corruption")
        
        # Log CUDA Graph setting
        perf_mode = "High Performance (CUDA Graphs ON)" if enable_cuda_graphs else "Memory Safe (CUDA Graphs OFF)"
        print(f"🔧 Higgs Audio mode: {perf_mode}")
        
        # Create the base HiggsAudioEngine but initialize manually to avoid circular dependency
        base_engine = HiggsAudioEngine()
        
        # Manually set up the engine properties to avoid calling initialize_engine
        base_engine.model_path = model_path
        base_engine.tokenizer_path = tokenizer_path
        base_engine.device = device
        
        # Get smart paths using the engine's methods
        model_path_for_engine = base_engine._get_smart_model_path(model_path)
        tokenizer_path_for_engine = base_engine._get_smart_tokenizer_path(tokenizer_path)
        
        # Configure CUDA Graph settings based on toggle
        if enable_cuda_graphs:
            print(f"🚀 Loading Higgs Audio model directly on {device} for optimal performance...")
            cache_type = "StaticCache"
            kv_cache_lengths = [1024, 2048, 4096]  # StaticCache for CUDA Graph optimization
        else:
            print(f"🚀 Loading Higgs Audio model directly on {device} (CUDA Graphs disabled for memory safety)...")
            cache_type = "DynamicCache" 
            # Still need cache buckets but they won't use StaticCache
            kv_cache_lengths = [1024, 2048, 4096]  # Same sizes but will use DynamicCache
            
            # Disable CUDA Graph compilation at environment level
            import os
            os.environ['PYTORCH_DISABLE_CUDA_GRAPH'] = '1'
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
        
        serve_engine = HiggsAudioServeEngine(
            model_name_or_path=model_path_for_engine,
            audio_tokenizer_name_or_path=tokenizer_path_for_engine,
            device=device,
            kv_cache_lengths=kv_cache_lengths,
            enable_cuda_graphs=enable_cuda_graphs
        )
        
        # Settings are already stored by the constructor
        
        print(f"✅ Higgs Audio model loaded directly on {device}")
        
        # Set the serve engine on the base engine
        base_engine.engine = serve_engine
        
        # Wrap with stateless wrapper for ComfyUI model management compatibility
        stateless_wrapper = create_stateless_higgs_wrapper(base_engine)
        
        print(f"📦 Higgs Audio engine ready (via unified interface)")
        return stateless_wrapper
    
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


# Auto-initialize on import
initialize_all_factories()