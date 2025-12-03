"""
Unified model loading utility supporting ModelScope, HuggingFace and local path loading
"""
import os
import sys
import logging
import threading
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add step_audio_editx_impl to sys.path so internal modules can import each other
_impl_dir = os.path.dirname(os.path.abspath(__file__))
if _impl_dir not in sys.path:
    sys.path.insert(0, _impl_dir)

from funasr_detach import AutoModel

# Global cache for downloaded models to avoid repeated downloads
# Key: (model_path, source)
# Value: local_model_path
_model_download_cache = {}
_download_cache_lock = threading.Lock()


class ModelSource:
    """Model source enumeration"""
    MODELSCOPE = "modelscope"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AUTO = "auto"  # Auto-detect


class UnifiedModelLoader:
    """Unified model loader"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _cached_snapshot_download(self, model_path: str, source: str, **kwargs) -> str:
        """
        Cached version of snapshot_download to avoid repeated downloads

        Args:
            model_path: Model path or ID to download
            source: Model source ('modelscope' or 'huggingface')
            **kwargs: Additional arguments for snapshot_download

        Returns:
            Local path to downloaded model
        """
        cache_key = (model_path, source, str(sorted(kwargs.items())))

        # Check cache first
        with _download_cache_lock:
            if cache_key in _model_download_cache:
                cached_path = _model_download_cache[cache_key]
                self.logger.info(f"Using cached download for {model_path} from {source}: {cached_path}")
                return cached_path

        # Cache miss, need to download
        if source == ModelSource.MODELSCOPE:
            from modelscope.hub.snapshot_download import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        elif source == ModelSource.HUGGINGFACE:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source for cached download: {source}")

        # Cache the result
        with _download_cache_lock:
            _model_download_cache[cache_key] = local_path

        self.logger.info(f"Downloaded and cached {model_path} from {source}: {local_path}")
        return local_path

    def detect_model_source(self, model_path: str) -> str:
        """
        Automatically detect model source

        Args:
            model_path: Model path or ID

        Returns:
            Model source type
        """
        # Local path detection
        if os.path.exists(model_path) or os.path.isabs(model_path):
            return ModelSource.LOCAL

        # ModelScope format detection (usually includes username/model_name)
        if "/" in model_path and not model_path.startswith("http"):
            # If contains modelscope keyword or is known modelscope format
            if "modelscope" in model_path.lower() or self._is_modelscope_format(model_path):
                return ModelSource.MODELSCOPE
            else:
                # Default to HuggingFace
                return ModelSource.HUGGINGFACE

        return ModelSource.LOCAL

    def _is_modelscope_format(self, model_path: str) -> bool:
        """Detect if it's ModelScope format model ID"""
        # Can be judged according to known ModelScope model ID formats
        # For example: iic/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online
        modelscope_patterns = []
        return any(pattern in model_path for pattern in modelscope_patterns)

    def _prepare_quantization_config(self, quantization_config: Optional[str], torch_dtype: Optional[torch.dtype] = None) -> Tuple[Dict[str, Any], bool]:
        """
        Prepare quantization configuration for model loading

        Args:
            quantization_config: Quantization type ('int4', 'int8', or None)
            torch_dtype: PyTorch data type for compute operations

        Returns:
            Tuple of (quantization parameters dict, should_set_torch_dtype)
        """
        if not quantization_config:
            return {}, True

        quantization_config = quantization_config.lower()

        if quantization_config == "int8":
            # Use user-specified torch_dtype for compute, default to bfloat16
            compute_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
            self.logger.info(f"🔧 INT8 quantization: using {compute_dtype} for compute operations")

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
            return {
                "quantization_config": bnb_config
            }, False  # INT8 quantization handles data types automatically, don't set torch_dtype
        elif quantization_config == "int4":
            # Use user-specified torch_dtype for compute, default to bfloat16
            compute_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
            self.logger.info(f"🔧 INT4 quantization: using {compute_dtype} for compute operations")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            return {
                "quantization_config": bnb_config
            }, False  # INT4 quantization handles torch_dtype internally, don't set it again
        elif quantization_config == "awq-4bit":
            # AWQ 4-bit quantization - requires special model path handling
            self.logger.info(f"🔧 AWQ 4-bit quantization enabled")
            return {}, True  # No special quantization_config, but allow torch_dtype setting
        else:
            raise ValueError(f"Unsupported quantization config: {quantization_config}. Supported: 'int4', 'int8', 'awq-4bit'")

    def load_transformers_model(
        self,
        model_path: str,
        source: str = ModelSource.AUTO,
        quantization_config: Optional[str] = None,
        **kwargs
    ) -> Tuple:
        """
        Load Transformers model (for StepAudioTTS)

        Args:
            model_path: Model path or ID
            source: Model source, auto means auto-detect
            quantization_config: Quantization configuration ('int4', 'int8', or None for no quantization)
            **kwargs: Other parameters (torch_dtype, device_map, etc.)

        Returns:
            (model, tokenizer) tuple
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)

        self.logger.info(f"Loading Transformers model from {source}: {model_path}")
        if quantization_config:
            self.logger.info(f"🔧 {quantization_config.upper()} quantization enabled")

        # Prepare quantization configuration
        quantization_kwargs, should_set_torch_dtype = self._prepare_quantization_config(quantization_config, kwargs.get("torch_dtype"))

        try:
            if source == ModelSource.LOCAL:
                # Local loading
                load_kwargs = {
                    "device_map": kwargs.get("device_map", "auto"),
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "attn_implementation": "eager"  # Step1ForCausalLM only supports eager mode
                }

                # Add quantization configuration if specified
                load_kwargs.update(quantization_kwargs)

                # Add torch_dtype based on quantization requirements
                if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                    load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

            elif source == ModelSource.MODELSCOPE:
                # Load from ModelScope
                from modelscope import AutoModelForCausalLM as MSAutoModelForCausalLM
                from modelscope import AutoTokenizer as MSAutoTokenizer
                model_path = self._cached_snapshot_download(model_path, ModelSource.MODELSCOPE)

                load_kwargs = {
                    "device_map": kwargs.get("device_map", "auto"),
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "attn_implementation": "eager"  # Step1ForCausalLM only supports eager mode
                }

                # Add quantization configuration if specified
                load_kwargs.update(quantization_kwargs)

                # Add torch_dtype based on quantization requirements
                if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                    load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

            elif source == ModelSource.HUGGINGFACE:
                model_path = self._cached_snapshot_download(model_path, ModelSource.HUGGINGFACE)

                # Load from HuggingFace
                load_kwargs = {
                    "device_map": kwargs.get("device_map", "auto"),
                    "attn_implementation": "sdpa",  # CRITICAL: Use SDPA for proper generation
                    "trust_remote_code": True,
                    "local_files_only": True
                }

                # Add quantization configuration if specified
                # Add quantization configuration if specified
                load_kwargs.update(quantization_kwargs)

                # Add torch_dtype based on quantization requirements
                if should_set_torch_dtype and kwargs.get("torch_dtype") is not None:
                    load_kwargs["torch_dtype"] = kwargs.get("torch_dtype")

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

            else:
                raise ValueError(f"Unsupported model source: {source}")

            self.logger.info(f"Successfully loaded model from {source}")
            return model, tokenizer, model_path

        except Exception as e:
            self.logger.error(f"Failed to load model from {source}: {e}")
            raise

    def load_funasr_model(
        self,
        repo_path: str,
        model_path: str,
        source: str = ModelSource.AUTO,
        **kwargs
    ) -> AutoModel:
        """
        Load FunASR model (for StepAudioTokenizer)

        Args:
            model_path: Model path or ID
            source: Model source, auto means auto-detect
            **kwargs: Other parameters

        Returns:
            FunASR AutoModel instance
        """
        if source == ModelSource.AUTO:
            source = self.detect_model_source(model_path)
            
        self.logger.info(f"Loading FunASR model from {source}: {model_path}")

        try:
            # Extract model_revision to avoid duplicate passing
            model_revision = kwargs.pop("model_revision", "main")

            # Map ModelSource to model_hub parameter
            if source == ModelSource.LOCAL:
                model_hub = "local"
            elif source == ModelSource.MODELSCOPE:
                model_hub = "ms"
            elif source == ModelSource.HUGGINGFACE:
                model_hub = "hf"
            else:
                raise ValueError(f"Unsupported model source: {source}")

            # Use unified download_model for all cases
            model = AutoModel(
                repo_path=repo_path,
                model=model_path,
                model_hub=model_hub,
                model_revision=model_revision,
                **kwargs
            )

            self.logger.info(f"Successfully loaded FunASR model from {source}")
            return model

        except Exception as e:
            self.logger.error(f"Failed to load FunASR model from {source}: {e}")
            raise

    def resolve_model_path(
        self,
        base_path: str,
        model_name: str,
        source: str = ModelSource.AUTO
    ) -> str:
        """
        Resolve model path

        Args:
            base_path: Base path
            model_name: Model name
            source: Model source

        Returns:
            Resolved model path
        """
        if source == ModelSource.AUTO:
            # First check local path
            local_path = os.path.join(base_path, model_name)
            if os.path.exists(local_path):
                return local_path

            # If local doesn't exist, return model name for online download
            return model_name

        elif source == ModelSource.LOCAL:
            return os.path.join(base_path, model_name)

        else:
            # For online sources, directly return model name/ID
            return model_name


# Global instance
model_loader = UnifiedModelLoader()