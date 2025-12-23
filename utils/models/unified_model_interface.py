"""
Unified Model Interface for TTS Audio Suite

This module provides a standardized interface for all TTS engines to use
ComfyUI's model management system. All engines (ChatterBox, F5-TTS, Higgs Audio, 
RVC, etc.) should use this interface instead of direct caching.
"""

import torch
from typing import Any, Dict, Optional, Callable, Union
from pathlib import Path

from utils.models.comfyui_model_wrapper import tts_model_manager, ModelInfo
from utils.models.factory_config import ModelLoadConfig


class UnifiedModelInterface:
    """
    Unified interface for all TTS engine model loading.
    
    This replaces all engine-specific caching systems with a single
    ComfyUI-integrated approach.
    """
    
    def __init__(self):
        """Initialize the unified interface"""
        self._model_factories: Dict[str, Callable] = {}
        self._pytorch_warning_shown = False
        
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
    
    def _check_pytorch_consistency(self) -> None:
        """Check for PyTorch/TorchAudio/TorchVision consistency and warn if mixed CUDA/CPU"""
        if self._pytorch_warning_shown:
            return
            
        try:
            import torch
            import torchaudio
            import torchvision
            
            # Check PyTorch version info
            torch_version = getattr(torch, '__version__', 'unknown')
            torchaudio_version = getattr(torchaudio, '__version__', 'unknown')  
            torchvision_version = getattr(torchvision, '__version__', 'unknown')
            
            # Detect CUDA vs CPU variants
            torch_cuda = '+cu' in torch_version
            torch_cpu = '+cpu' in torch_version
            torchaudio_cpu = '+cpu' in torchaudio_version or (not '+cu' in torchaudio_version and not torch_cpu)
            torchvision_cpu = '+cpu' in torchvision_version or (not '+cu' in torchvision_version and not torch_cpu)
            
            warnings = []
            
            # Check for performance-killing mixed installations
            if torch_cuda and torchaudio_cpu:
                warnings.append("❌ CRITICAL PERFORMANCE WARNING: GPU PyTorch + CPU TorchAudio detected!")
                warnings.append(f"   PyTorch: {torch_version} | TorchAudio: {torchaudio_version}")
                warnings.append("   This causes major slowdowns and VRAM spikes in TTS generation.")
                warnings.append("   Fix: pip uninstall torchaudio && pip install torchaudio --index-url https://download.pytorch.org/whl/cu129")
                
            if torch_cuda and torchvision_cpu:
                warnings.append("❌ PERFORMANCE WARNING: GPU PyTorch + CPU TorchVision detected!")
                warnings.append(f"   PyTorch: {torch_version} | TorchVision: {torchvision_version}")
                warnings.append("   Fix: pip uninstall torchvision && pip install torchvision --index-url https://download.pytorch.org/whl/cu129")
            
            if warnings:
                print("\n" + "="*80)
                print("🔥 TTS AUDIO SUITE - PYTORCH INSTALLATION WARNING")
                print("="*80)
                for warning in warnings:
                    print(warning)
                print("="*80 + "\n")
                
            self._pytorch_warning_shown = True
            
        except ImportError as e:
            # PyTorch not available - this will be caught elsewhere
            pass
        except Exception as e:
            # Don't let consistency check break model loading
            print(f"⚠️ PyTorch consistency check failed: {e}")
            self._pytorch_warning_shown = True

    def load_model(self, config: ModelLoadConfig, force_reload: bool = False) -> Any:
        """
        Load a model using the unified interface.

        Args:
            config: Model configuration (ModelLoadConfig)
            force_reload: Force reload even if cached

        Returns:
            The loaded model (unwrapped from ComfyUI wrapper)

        Raises:
            ValueError: If no factory is registered for the engine/model type
            RuntimeError: If model loading fails
        """
        # Check PyTorch consistency on first model load
        self._check_pytorch_consistency()

        # Generate unique cache key
        cache_key = self._generate_cache_key(config)

        # Check if force reload is needed
        if force_reload:
            tts_model_manager.remove_model(cache_key)

        # Get existing model if cached
        wrapper = tts_model_manager.get_model(cache_key)
        if wrapper is not None:
            # Check if wrapper's model was deleted (e.g., by ComfyUI auto-unload for bitsandbytes models)
            if hasattr(wrapper, 'is_dead') and wrapper.is_dead():
                print(f"⚠️ Cached model wrapper is dead (model was deleted), forcing reload...")
                tts_model_manager.remove_model(cache_key)
                wrapper = None  # Fall through to reload
            else:
                # Ensure model is on correct device
                from utils.device import resolve_torch_device
                target_device = resolve_torch_device(config.device)

                # Resolve wrapper's current device for consistent comparison
                # wrapper.current_device might be string or torch.device object
                wrapper_device_resolved = resolve_torch_device(
                    wrapper.current_device if isinstance(wrapper.current_device, str)
                    else str(wrapper.current_device)
                )

                # Reload if device mismatch (handles both explicit devices and "auto" resolution)
                if wrapper_device_resolved != target_device:
                    wrapper.model_load(target_device)
                return wrapper

        # Find appropriate factory
        factory_key = f"{config.engine_name}_{config.model_type}"
        if factory_key not in self._model_factories:
            raise ValueError(f"No model factory registered for {factory_key}")

        factory_func = self._model_factories[factory_key]

        # Use ComfyUI model manager to load (pass config to factory)
        wrapper = tts_model_manager.load_model(
            model_factory_func=factory_func,
            model_key=cache_key,
            model_type=config.model_type,
            engine=config.engine_name,
            device=config.device,
            force_reload=force_reload,
            config=config  # Pass entire config to factory
        )

        return wrapper
    
    def unload_model(self, config: ModelLoadConfig) -> bool:
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

    def get_cached_model(self, engine_name: str, model_type: str = "tts"):
        """
        Get cached model by engine name and type.

        Args:
            engine_name: Engine name (e.g., "vibevoice", "step_audio_editx")
            model_type: Model type (default: "tts")

        Returns:
            Cached model wrapper or None if not found
        """
        # Search through cache for matching engine
        from utils.models.manager import tts_model_manager
        for cache_key in list(tts_model_manager._model_cache.keys()):
            if cache_key.startswith(f"{engine_name}_{model_type}_"):
                return tts_model_manager.get_model(cache_key)
        return None

    def _generate_cache_key(self, config: ModelLoadConfig) -> str:
        """Generate unique cache key for model configuration"""
        components = [
            config.engine_name,
            config.model_type,
            config.model_name,
            config.device,
        ]

        # CRITICAL FIX: ChatterBox Official 23-Lang is multilingual - language shouldn't affect cache
        # Only single-language models should include language in the key
        if config.engine_name != "chatterbox_official_23lang":
            components.append(config.language or "default")

        # Add path/repo info if available
        if config.model_path:
            components.append(f"path:{Path(config.model_path).name}")
        elif config.repo_id:
            components.append(f"repo:{config.repo_id}")

        # Add additional_params to ensure attention_mode, quantization, etc. trigger reloads
        if config.additional_params:
            # Sort params for consistent cache keys
            sorted_params = sorted(config.additional_params.items())
            params_str = "_".join([f"{k}:{v}" for k, v in sorted_params])
            components.append(f"params:{params_str}")

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
    config = ModelLoadConfig(
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
    config = ModelLoadConfig(
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
    config = ModelLoadConfig(
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
    """Register ChatterBox model factories with universal component mixing logic"""
    
    def load_chatterbox_with_mixing(target_class, device, language, model_path):
        """Universal ChatterBox loading with component mixing fallback"""
        from engines.chatterbox.language_models import find_local_model_path
        
        # Ensure target_class is available before attempting to use it
        if target_class is None:
            raise RuntimeError(f"ChatterBox class not available - cannot load model")
        
        # Try provided path first
        if model_path:
            try:
                return target_class.from_local(model_path, device)
            except Exception as e:
                print(f"⚠️ Failed to load from provided path {model_path}: {e}")
                # Continue to fallback logic
        
        # Try language-specific local model
        try:
            local_path = find_local_model_path(language)
            if local_path:
                return target_class.from_local(local_path, device)
        except Exception as e:
            print(f"⚠️ Failed to load local {language} model: {e}")
        
        # Try HuggingFace download for requested language
        try:
            return target_class.from_pretrained(device, language)
        except Exception as e:
            error_str = str(e)
            print(f"⚠️ Failed to download {language} model: {e}")
            
            # Check for 401 authorization errors (gated/private repos)
            if "401" in error_str or "Unauthorized" in error_str:
                print(f"🔒 {language} model requires manual download due to authentication requirements")
                print(f"   Please visit the model repository and download manually to use this model")
                
                # For German variants, fall back to base German model instead of English
                if language.startswith("German (") and language != "German":
                    print(f"🔄 Falling back to base German model for ChatterBox")
                    # Try the complete fallback sequence for German
                    try:
                        german_local = find_local_model_path("German")
                        if german_local:
                            return target_class.from_local(german_local, device)
                    except Exception as local_error:
                        print(f"⚠️ German local model failed: {local_error}")
                        
                    # If local German fails, try downloading German
                    try:
                        print(f"📥 Attempting to download base German model")
                        return target_class.from_pretrained(device, "German")
                    except Exception as german_error:
                        print(f"⚠️ German download also failed: {german_error}")
                        # Continue to English fallback
            
            # Final fallback: English model if not already trying English
            if language != "English":
                print(f"🔄 Falling back to English model for ChatterBox")
                try:
                    # Try English local first
                    english_local = find_local_model_path("English")
                    if english_local:
                        return target_class.from_local(english_local, device)
                    else:
                        return target_class.from_pretrained(device, "English")
                except Exception as english_error:
                    print(f"❌ English fallback also failed: {english_error}")
                    raise e  # Raise original error
            else:
                raise e  # Already trying English, fail
    
    def chatterbox_tts_factory(config: ModelLoadConfig):
        try:
            from engines.chatterbox.tts import ChatterboxTTS
        except ImportError:
            ChatterboxTTS = None

        if ChatterboxTTS is None:
            raise RuntimeError("ChatterboxTTS not available - check installation")

        return load_chatterbox_with_mixing(ChatterboxTTS, config.device, config.language or "English", config.model_path)

    def chatterbox_vc_factory(config: ModelLoadConfig):
        try:
            from engines.chatterbox.vc import ChatterboxVC
            from engines.chatterbox.language_models import supports_voice_conversion, get_vc_supported_languages
        except ImportError:
            ChatterboxVC = None
            supports_voice_conversion = None

        if ChatterboxVC is None:
            raise RuntimeError("ChatterboxVC not available - check installation or add bundled version")

        language = config.language or "English"

        # Check if language supports VC before attempting to load
        if supports_voice_conversion and not supports_voice_conversion(language):
            print(f"❌ {language} model does not support voice conversion")
            print(f"   Voice conversion requires s3gen component which is missing from this model")
            print(f"   Try English model (confirmed working) or German models (tested working)")
            print(f"   Other language models may not have VC components")

            raise RuntimeError(f"Voice conversion not supported for {language} model. "
                             f"Try English or German models instead.")

        return load_chatterbox_with_mixing(ChatterboxVC, config.device, language, config.model_path)
    
    unified_model_interface.register_model_factory("chatterbox", "tts", chatterbox_tts_factory)
    unified_model_interface.register_model_factory("chatterbox", "vc", chatterbox_vc_factory)


def register_f5tts_factory():
    """Register F5-TTS model factories"""
    def f5tts_factory(config: ModelLoadConfig):
        from engines.f5tts.f5tts import ChatterBoxF5TTS
        import os
        import folder_paths

        model_name = config.model_name or "F5TTS_Base"

        # Try local first, then HuggingFace
        try:
            # Check for local models
            search_paths = [
                os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", model_name),
                os.path.join(folder_paths.models_dir, "F5-TTS", model_name),  # Legacy
                os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS", model_name)  # Legacy
            ]

            local_path = None
            for path in search_paths:
                if os.path.exists(path):
                    local_path = path
                    break

            if local_path:
                print(f"📁 Using local F5-TTS model: {local_path}")
                return ChatterBoxF5TTS.from_local(local_path, config.device, model_name)
            else:
                print(f"📥 Loading F5-TTS model '{model_name}' from HuggingFace")
                return ChatterBoxF5TTS.from_pretrained(config.device, model_name)

        except Exception as e:
            raise RuntimeError(f"Failed to load F5-TTS model '{model_name}': {e}")

    unified_model_interface.register_model_factory("f5tts", "tts", f5tts_factory)


def register_higgs_audio_factory():
    """Register Higgs Audio model factories with stateless wrapper for ComfyUI integration"""
    def higgs_audio_factory(config: ModelLoadConfig):
        from engines.higgs_audio.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        from engines.higgs_audio.higgs_audio import HiggsAudioEngine
        from engines.higgs_audio.stateless_wrapper import create_stateless_higgs_wrapper

        model_name = config.model_name
        device = config.device or "cuda"
        model_path = config.model_path or model_name
        tokenizer_path = config.additional_params.get("tokenizer_path") if config.additional_params else None
        enable_cuda_graphs = config.additional_params.get("enable_cuda_graphs", True) if config.additional_params else True

        # Check if we need to force recreate due to CUDA Graph corruption
        force_recreate = config.additional_params.get("_force_recreate", False) if config.additional_params else False
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
    def rvc_factory(config: ModelLoadConfig):
        """Load RVC model and wrap it for ComfyUI management"""
        from engines.rvc.impl.vc_infer_pipeline import get_vc
        from engines.rvc.impl.config import config as rvc_config
        from engines.rvc.rvc_engine import RVCModelWrapper

        model_path = config.model_path
        index_path = config.additional_params.get("index_path") if config.additional_params else None
        device = config.device or "cuda"

        if not model_path:
            raise ValueError("RVC factory requires model_path")

        # Load the RVC model using the actual implementation
        model_data = get_vc(model_path, index_path, rvc_config, device)

        # Wrap the model_data dict so it has .to() method for device management
        wrapped_model = RVCModelWrapper(model_data, device)

        return wrapped_model

    def hubert_factory(config: ModelLoadConfig):
        from engines.rvc.impl.lib.model_utils import load_hubert
        model_path = config.model_path
        config_dict = config.additional_params.get("config", {}) if config.additional_params else {}
        return load_hubert(model_path, config_dict)

    def mdx23c_separator_factory(config: ModelLoadConfig):
        from engines.rvc.impl.lib.mdx23c import MDX23C
        model_path = config.model_path
        device = config.device or "cuda"
        return MDX23C(model_path, device)

    def demucs_separator_factory(config: ModelLoadConfig):
        from engines.rvc.impl.lib.uvr5_pack.demucs.pretrained import get_model
        model_name = config.model_name or "htdemucs"
        device = config.device or "cuda"
        return get_model(model_name).to(device)

    unified_model_interface.register_model_factory("rvc", "vc", rvc_factory)
    unified_model_interface.register_model_factory("rvc", "hubert", hubert_factory)
    unified_model_interface.register_model_factory("rvc", "separator_mdx", mdx23c_separator_factory)
    unified_model_interface.register_model_factory("rvc", "separator_demucs", demucs_separator_factory)


def register_vibevoice_factory():
    """Register VibeVoice model factory"""
    def vibevoice_factory(config: ModelLoadConfig):
        """Factory for VibeVoice models with ComfyUI integration"""

        # Ensure accelerate is imported before VibeVoice engine
        try:
            import accelerate
            # (accelerate version check removed - only show on errors)
        except ImportError as e:
            print(f"⚠️ Unified Interface: accelerate not available: {e}")

        from engines.vibevoice_engine.vibevoice_engine import VibeVoiceEngine

        # Extract parameters
        model_name = config.model_name or "vibevoice-1.5B"
        device = config.device or "auto"
        attention_mode = config.additional_params.get("attention_mode", "auto") if config.additional_params else "auto"
        quantize_llm_4bit = config.additional_params.get("quantize_llm_4bit", False) if config.additional_params else False

        # Create engine instance
        engine = VibeVoiceEngine()

        # Initialize with parameters
        engine.initialize_engine(
            model_name=model_name,
            device=device,
            attention_mode=attention_mode,
            quantize_llm_4bit=quantize_llm_4bit
        )

        print(f"✅ VibeVoice model '{model_name}' loaded via unified interface")

        # Return the engine which contains both model and processor
        return engine

    unified_model_interface.register_model_factory("vibevoice", "tts", vibevoice_factory)


def register_index_tts_factory():
    """Register IndexTTS-2 model factory"""
    def index_tts_factory(config: ModelLoadConfig):
        """Factory for IndexTTS-2 models with ComfyUI integration"""
        import os
        import sys

        # Extract parameters
        model_path = config.model_path
        device = config.device or "auto"
        use_fp16 = config.additional_params.get("use_fp16", True) if config.additional_params else True
        use_cuda_kernel = config.additional_params.get("use_cuda_kernel", None) if config.additional_params else None
        use_deepspeed = config.additional_params.get("use_deepspeed", False) if config.additional_params else False
        use_torch_compile = config.additional_params.get("use_torch_compile", False) if config.additional_params else False
        low_vram = config.additional_params.get("low_vram", False) if config.additional_params else False
        
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f"IndexTTS-2 model not found at {model_path}. Auto-download should have been triggered earlier.")
        
        try:
            # Add bundled IndexTTS path to sys.path so internal imports work
            import sys
            bundled_path = os.path.join(os.path.dirname(__file__), "..", "..", "engines", "index_tts")
            bundled_path = os.path.abspath(bundled_path)
            if bundled_path not in sys.path:
                sys.path.insert(0, bundled_path)

            # Check for conflicting external IndexTTS installation
            try:
                import indextts
                external_path = indextts.__file__
                if ('site-packages' in external_path or
                    'conda' in external_path or
                    'pip' in external_path or
                    'dist-packages' in external_path):

                    raise ImportError(f"""
❌ External IndexTTS installation detected!

   Found at: {external_path}

   This conflicts with our bundled IndexTTS-2 engine and causes import errors.

   🔧 SOLUTION: Please uninstall the external version:
      pip uninstall indextts

   Then restart ComfyUI.

   Our bundled version has all required dependencies and will work perfectly.
""")
            except ImportError as e:
                if "External IndexTTS" in str(e):
                    raise e
                # No external installation found - this is good!
                pass
            except Exception:
                # Some other issue with indextts, proceed anyway
                pass

            # Import from our bundled IndexTTS engine
            from engines.index_tts.indextts.infer_v2 import IndexTTS2
            
            # Initialize IndexTTS-2 engine
            config_path = os.path.join(model_path, "config.yaml")

            # Verify config file exists after download
            if not os.path.exists(config_path):
                raise RuntimeError(f"IndexTTS-2 config.yaml not found at {config_path} even after download. Please check model integrity.")
            
            engine = IndexTTS2(
                cfg_path=config_path,
                model_dir=model_path,
                device=device,
                use_fp16=use_fp16 and device != "cpu",
                use_cuda_kernel=use_cuda_kernel,
                use_deepspeed=use_deepspeed,
                use_torch_compile=use_torch_compile,
                low_vram=low_vram
            )
            
            print(f"✅ IndexTTS-2 model loaded via unified interface on {device}")
            return engine
            
        except ImportError as e:
            raise ImportError(f"IndexTTS-2 dependencies not available. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load IndexTTS-2 model: {e}")
    
    unified_model_interface.register_model_factory("index_tts", "tts", index_tts_factory)

def register_chatterbox_23lang_factory():
    """Register ChatterBox Official 23-Lang model factory"""
    def chatterbox_23lang_factory(config: ModelLoadConfig):
        """Factory for ChatterBox Official 23-Lang models with ComfyUI integration"""
        from engines.chatterbox_official_23lang.tts import ChatterboxOfficial23LangTTS
        import folder_paths
        import os
        import torch

        # Extract parameters
        device = config.device or "auto"
        model_name = config.model_name or "Official 23-Lang"  # Always same model
        language = config.language or "english"  # This is the actual language to use
        model_version = config.additional_params.get("model_version", "v2") if config.additional_params else "v2"  # v1 or v2

        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get model directory path
        models_dir = folder_paths.models_dir
        ckpt_dir = os.path.join(models_dir, "TTS", "chatterbox_official_23lang", "Official 23-Lang")

        print(f"🌍 Loading ChatterBox Official 23-Lang model for {language} on {device}")
        print(f"📁 Using model directory: {ckpt_dir}")

        # Try local first, then use from_pretrained for auto-download if needed
        if os.path.exists(ckpt_dir):
            try:
                engine = ChatterboxOfficial23LangTTS.from_local(
                    ckpt_dir=ckpt_dir,
                    device=device,
                    model_name="Official 23-Lang",
                    model_version=model_version
                )
                print(f"✅ ChatterBox Official 23-Lang '{language}' loaded via unified interface")
                return engine
            except FileNotFoundError as e:
                print(f"⚠️ Local model incomplete: {e}")
                print(f"📥 Downloading missing {model_version} files...")

        # Use from_pretrained for auto-download
        engine = ChatterboxOfficial23LangTTS.from_pretrained(
            device=device,
            model_name="ChatterBox Official 23-Lang",
            model_version=model_version
        )

        print(f"✅ ChatterBox Official 23-Lang '{language}' loaded via unified interface")

        # Return the engine
        return engine
    
    unified_model_interface.register_model_factory("chatterbox_official_23lang", "tts", chatterbox_23lang_factory)


def register_step_audio_editx_factory():
    """Register Step Audio EditX model factory"""
    def step_audio_editx_factory(config: ModelLoadConfig):
        """Factory for Step Audio EditX models with ComfyUI integration"""
        import os
        import sys
        import torch

        # Extract parameters
        model_path = config.model_path
        device = config.device or "auto"
        torch_dtype_str = config.additional_params.get("torch_dtype", "bfloat16") if config.additional_params else "bfloat16"
        quantization = config.additional_params.get("quantization", None) if config.additional_params else None

        # Resolve model path using downloader (handles "local:" prefix and auto-download)
        from engines.step_audio_editx.step_audio_editx_downloader import StepAudioEditXDownloader
        downloader = StepAudioEditXDownloader()
        model_path = downloader.resolve_model_path(model_path)

        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f"Step Audio EditX model not found at {model_path}. Auto-download should have been triggered earlier.")

        try:
            # Add bundled Step Audio EditX path to sys.path so internal imports work
            bundled_path = os.path.join(os.path.dirname(__file__), "..", "..", "engines", "step_audio_editx")
            bundled_path = os.path.abspath(bundled_path)
            if bundled_path not in sys.path:
                sys.path.insert(0, bundled_path)

            # Import from bundled Step Audio EditX engine
            from engines.step_audio_editx.step_audio_editx_impl.tts import StepAudioTTS
            from engines.step_audio_editx.step_audio_editx_impl.tokenizer import StepAudioTokenizer
            from engines.step_audio_editx.step_audio_editx_impl.model_loader import ModelSource

            # Resolve torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            }
            torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

            # Resolve device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Resolve quantization config
            quantization_config = None
            if quantization and quantization != "none":
                quantization_config = quantization  # Will be 'int4', 'int8', etc.

            # Auto-download all required models if not present
            from engines.step_audio_editx.step_audio_editx_downloader import StepAudioEditXDownloader
            downloader = StepAudioEditXDownloader()
            base_path = os.path.dirname(model_path)

            # Check and download Step-Audio-EditX main model (includes tokenizer files)
            main_files = downloader.MODELS["Step-Audio-EditX"]["files"]
            if main_files and isinstance(main_files[0], dict):
                check_files = [f["local"] for f in main_files]
            else:
                check_files = main_files

            # Also add additional downloads to completeness check
            additional_downloads = downloader.MODELS["Step-Audio-EditX"].get("additional_downloads", [])
            for additional in additional_downloads:
                for file_dict in additional["files"]:
                    check_files.append(file_dict["local"])

            is_complete = downloader._is_model_complete(model_path, check_files)

            if not is_complete:
                print(f"📥 Step-Audio-EditX model incomplete, downloading...")
                downloader.download_model("Step-Audio-EditX")

            # Check and download FunASR model (required for tokenizer)
            funasr_path = os.path.join(base_path, "FunASR-Paraformer")
            funasr_files = downloader.MODELS["FunASR-Paraformer"]["files"]
            if not downloader._is_model_complete(funasr_path, funasr_files):
                print(f"📥 FunASR model not found or incomplete, downloading (one-time setup)...")
                downloader.download_model("FunASR-Paraformer")

            # CRITICAL: Verify CosyVoice model exists (required for audio synthesis)
            # CosyVoice is a subfolder of the main model and must be complete
            cosy_voice_path = os.path.join(model_path, "CosyVoice-300M-25Hz")
            cosy_voice_files = [
                "cosyvoice.yaml", "flow.pt", "hift.pt",
                "campplus.onnx", "speech_tokenizer_v1.onnx", "FLOW_VERSION"
            ]

            if not os.path.exists(cosy_voice_path):
                raise RuntimeError(
                    f"❌ CosyVoice model not found at: {cosy_voice_path}\n"
                    f"This subfolder should have been downloaded with Step-Audio-EditX.\n"
                    f"To fix: Delete the entire Step-Audio-EditX folder and re-run to trigger full download.\n"
                    f"Model path: {model_path}"
                )

            # Check CosyVoice completeness
            missing_files = []
            for file in cosy_voice_files:
                file_path = os.path.join(cosy_voice_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)

            if missing_files:
                raise RuntimeError(
                    f"❌ CosyVoice model incomplete. Missing files: {', '.join(missing_files)}\n"
                    f"Location: {cosy_voice_path}\n"
                    f"To fix: Delete the CosyVoice-300M-25Hz folder and re-run to trigger re-download."
                )

            # Note: cosyvoice.yaml uses HyperPyYAML custom tags (!new:, !ref, etc.)
            # so we cannot validate it with standard yaml.safe_load().
            # The actual validation happens when CosyVoice loads it with load_hyperpyyaml().

            # Initialize tokenizer (silent - errors will be raised if fails)
            tokenizer = StepAudioTokenizer(
                encoder_path=model_path,
                model_source=ModelSource.LOCAL  # Always use local for now
            )

            # Initialize TTS engine (silent - errors will be raised if fails)
            tts_engine = StepAudioTTS(
                model_path=model_path,
                audio_tokenizer=tokenizer,
                model_source=ModelSource.LOCAL,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device
            )

            # Return wrapped engine directly instead of using wrapper class
            # The adapter and nodes expect a wrapper with clone() and edit_single() methods
            # Create a simple wrapper that delegates to the raw TTS engine
            class StepAudioEditXEngineWrapper:
                """Simple wrapper providing clone() and edit_single() methods for the raw StepAudioTTS engine"""
                def __init__(self, tts_engine, tokenizer, device, torch_dtype, quantization, model_path):
                    self._tts_engine = tts_engine
                    self._tokenizer = tokenizer
                    self._model_config = None
                    self.device = device
                    self.torch_dtype = torch_dtype
                    self.quantization = quantization
                    self.model_dir = model_path

                def clone(self, prompt_wav_path, prompt_text, target_text, temperature=0.7, do_sample=True, max_new_tokens=8192, progress_bar=None):
                    """Delegate clone to raw TTS engine"""
                    return self._tts_engine.clone(
                        prompt_wav_path=prompt_wav_path,
                        prompt_text=prompt_text,
                        target_text=target_text,
                        progress_bar=progress_bar,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample
                    )

                def edit_single(self, input_audio_path, audio_text, edit_type, edit_info=None, text=None, progress_bar=None, max_new_tokens=8192, temperature=0.7, do_sample=True):
                    """Delegate edit_single to raw TTS engine"""
                    audio_tensor, sample_rate = self._tts_engine.edit(
                        input_audio_path=input_audio_path,
                        audio_text=audio_text,
                        edit_type=edit_type,
                        edit_info=edit_info,
                        text=text,
                        progress_bar=progress_bar,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample
                    )
                    # Ensure output is [1, samples] format (2D)
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    elif audio_tensor.dim() == 3:
                        audio_tensor = audio_tensor.squeeze(0)
                    return audio_tensor

                def edit(self, input_audio_path, audio_text, edit_type, edit_info=None, text=None, progress_bar=None, max_new_tokens=8192, temperature=0.7, do_sample=True):
                    """Alias for edit_single for compatibility"""
                    return self.edit_single(input_audio_path, audio_text, edit_type, edit_info, text, progress_bar, max_new_tokens, temperature, do_sample)

                def get_sample_rate(self):
                    """Get sample rate from raw engine"""
                    return 24000  # CosyVoice native sample rate

                def to(self, device):
                    """Move model to device"""
                    self.device = device
                    if hasattr(self._tts_engine, 'llm') and hasattr(self._tts_engine.llm, 'to'):
                        try:
                            self._tts_engine.llm.to(device)
                        except:
                            pass  # Quantized models can't be moved
                    if hasattr(self._tts_engine, 'cosy_model') and hasattr(self._tts_engine.cosy_model, 'to'):
                        try:
                            self._tts_engine.cosy_model.to(device)
                        except:
                            pass

            wrapper = StepAudioEditXEngineWrapper(tts_engine, tokenizer, device, torch_dtype, quantization, model_path)
            print(f"✅ Step Audio EditX model loaded via unified interface on {device}")
            return wrapper

        except ImportError as e:
            raise ImportError(f"Step Audio EditX dependencies not available. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Step Audio EditX model: {e}")

    unified_model_interface.register_model_factory("step_audio_editx", "tts", step_audio_editx_factory)


def register_cosyvoice_factory():
    """Register CosyVoice3 model factory"""
    def cosyvoice_factory(config: ModelLoadConfig):
        """Factory for CosyVoice3 models with ComfyUI integration"""
        import os
        import sys
        
        # Extract parameters
        model_path = config.model_path
        device = config.device or "auto"
        use_fp16 = config.additional_params.get("use_fp16", True) if config.additional_params else True
        load_trt = config.additional_params.get("load_trt", False) if config.additional_params else False
        load_vllm = config.additional_params.get("load_vllm", False) if config.additional_params else False
        
        try:
            # Import CosyVoice from bundled location
            # First, add the bundled CosyVoice path to sys.path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # TTS-Audio-Suite root
            bundled_cosyvoice_path = os.path.join(current_dir, "engines", "cosyvoice", "impl")
            
            if os.path.exists(bundled_cosyvoice_path):
                # Add bundled path to sys.path if not already there
                if bundled_cosyvoice_path not in sys.path:
                    sys.path.insert(0, bundled_cosyvoice_path)
                    print(f"📦 Using bundled CosyVoice library from {bundled_cosyvoice_path}")
                
                # Also add third_party/Matcha-TTS to path (required by CosyVoice)
                matcha_path = os.path.join(bundled_cosyvoice_path, "third_party", "Matcha-TTS")
                if os.path.exists(matcha_path) and matcha_path not in sys.path:
                    sys.path.insert(0, matcha_path)
                
                try:
                    from cosyvoice.cli.cosyvoice import CosyVoice3
                except ImportError as bundle_error:
                    raise ImportError(
                        f"Failed to import bundled CosyVoice: {bundle_error}\n"
                        "The bundled library may have missing dependencies.\n"
                        "Try: pip install hyperpyyaml conformer onnxruntime"
                    )
            else:
                # Fallback to system installation
                try:
                    from cosyvoice.cli.cosyvoice import CosyVoice3
                except ImportError:
                    raise ImportError(
                        "CosyVoice library not found. The bundled library is missing.\n"
                        "Please reinstall TTS Audio Suite or manually install CosyVoice:\n"
                        "1. Clone: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
                        "2. Install: cd CosyVoice && pip install -r requirements.txt\n"
                        "3. Restart ComfyUI"
                    )
            
            # Verify model path
            if not model_path:
                raise RuntimeError("CosyVoice3 model path not provided")
            
            if not os.path.exists(model_path):
                # Try auto-download
                from engines.cosyvoice.cosyvoice_downloader import cosyvoice_downloader
                model_path = cosyvoice_downloader.download_model()
            
            # Check for config file (support cosyvoice.yaml, cosyvoice2.yaml, or cosyvoice3.yaml)
            config_found = False
            for config_name in ["cosyvoice3.yaml", "cosyvoice2.yaml", "cosyvoice.yaml"]:
                config_path = os.path.join(model_path, config_name)
                if os.path.exists(config_path):
                    config_found = True
                    break
            
            if not config_found:
                raise RuntimeError(
                    f"CosyVoice config not found in {model_path}. "
                    f"Please ensure the model is correctly downloaded."
                )
            
            # Use AutoModel which automatically detects the correct model class
            # This matches the official usage pattern
            from cosyvoice.cli.cosyvoice import AutoModel
            
            print(f"🔄 Loading CosyVoice model from {model_path}...")
            
            # Handle vLLM gracefully - if not installed, just skip it
            actual_load_vllm = False
            if load_vllm:
                try:
                    import vllm
                    actual_load_vllm = True
                except ImportError:
                    print("⚠️ vLLM not installed, skipping vLLM optimization")
                    actual_load_vllm = False
            
            engine = AutoModel(
                model_dir=model_path,
                load_trt=load_trt,
                load_vllm=actual_load_vllm,
                fp16=use_fp16
            )
            
            print(f"✅ CosyVoice model loaded via unified interface on {device}")
            return engine
            
        except ImportError as e:
            raise ImportError(f"CosyVoice3 dependencies not available. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CosyVoice3 model: {e}")
    
    unified_model_interface.register_model_factory("cosyvoice", "tts", cosyvoice_factory)



def initialize_all_factories():
    """Initialize all model factories"""
    register_chatterbox_factory()
    register_chatterbox_23lang_factory()
    register_f5tts_factory()
    register_step_audio_editx_factory() 
    register_higgs_audio_factory()
    register_rvc_factory()
    register_vibevoice_factory()
    register_index_tts_factory()
    register_cosyvoice_factory()


# Auto-initialize on import
initialize_all_factories()


# Standalone helper functions for adapters
def get_cached_model(engine_name: str, model_type: str = "tts"):
    """
    Get cached model by engine name and type.

    Args:
        engine_name: Engine name (e.g., "vibevoice", "step_audio_editx")
        model_type: Model type (default: "tts")

    Returns:
        Cached model wrapper or None if not found
    """
    interface = UnifiedModelInterface()
    return interface.get_cached_model(engine_name, model_type)