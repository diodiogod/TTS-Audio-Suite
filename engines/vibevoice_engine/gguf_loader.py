"""
VibeVoice GGUF Loader - Load GGUF format VibeVoice models
Based on ComfyUI-GGUF reference implementation
"""

import os
import sys
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logger = logging.getLogger("VibeVoice-GGUF")

class VibeVoiceGGUFLoader:
    """Load GGUF format VibeVoice models and convert to standard PyTorch tensors"""
    
    def __init__(self):
        """Initialize GGUF loader"""
        self._gguf_available = None
    
    def _check_gguf_support(self) -> bool:
        """Check if GGUF loading is supported"""
        if self._gguf_available is not None:
            return self._gguf_available
        
        try:
            # Check if gguf package is available and load our utils
            import gguf
            from .gguf_utils import is_quantized, dequantize_tensor, load_gguf_state_dict
            
            logger.info("✅ gguf package and utilities loaded")
            self._gguf_available = True
            return True
            
        except ImportError:
            logger.error("❌ gguf package not installed")
            logger.error("💡 To use GGUF models, install with:")
            logger.error("   pip install gguf")
            logger.error("🔄 Or use DevParker's quantized models instead (no extra dependencies)")
            self._gguf_available = False
            return False
    
    def load_gguf_model(self, gguf_path: str, config_path: Optional[str] = None, keep_quantized: bool = True, skip_config: bool = False) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """
        Load GGUF model with optional quantization preservation for VRAM savings.
        
        Args:
            gguf_path: Path to .gguf file
            config_path: Optional path to config.json (will try to find automatically)
            keep_quantized: If True, keep quantized tensors (VRAM savings), if False, dequantize all (compatibility)
            skip_config: If True, skip config.json loading and use tensor-based detection
            
        Returns:
            Tuple of (state_dict, config) where state_dict contains tensors (quantized or dequantized)
        """
        if not self._check_gguf_support():
            raise RuntimeError("GGUF support not available. Install with: pip install gguf")
        
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        
        logger.info(f"🔄 Loading GGUF model: {gguf_path}")
        
        try:
            # Load GGUF state dict using our complete implementation
            from .gguf_utils import is_quantized, dequantize_tensor, load_gguf_state_dict
            
            raw_state_dict = load_gguf_state_dict(gguf_path)
            logger.info(f"📊 Loaded {len(raw_state_dict)} tensors from GGUF")
            
            # Process GGUF tensors based on keep_quantized setting
            state_dict = {}
            quantized_count = 0
            dequantized_count = 0
            kept_quantized_count = 0
            
            for key, tensor in raw_state_dict.items():
                if is_quantized(tensor):
                    quantized_count += 1
                    if keep_quantized:
                        # Keep quantized for VRAM savings - requires custom ops for inference
                        state_dict[key] = tensor  # Keep as GGMLTensor
                        kept_quantized_count += 1
                    else:
                        # Dequantize for compatibility
                        dequantized = dequantize_tensor(tensor, dtype=torch.bfloat16)
                        state_dict[key] = dequantized
                        dequantized_count += 1
                else:
                    # Already a standard tensor
                    state_dict[key] = tensor
            
            if keep_quantized and kept_quantized_count > 0:
                logger.info(f"✅ GGUF model loaded: {kept_quantized_count} tensors kept quantized (VRAM savings), {quantized_count-kept_quantized_count} dequantized, {len(state_dict)} total")
            else:
                logger.info(f"✅ GGUF model loaded: {dequantized_count} quantized tensors dequantized, {len(state_dict)} total tensors")
            
            # Try to load config.json (unless skipped for tensor-based detection)
            config = None
            if not skip_config:
                if config_path and os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"📋 Loaded config from: {config_path}")
                else:
                    # Try to find config in same directory as GGUF file
                    gguf_dir = os.path.dirname(gguf_path)
                    auto_config_path = os.path.join(gguf_dir, "config.json")
                    if os.path.exists(auto_config_path):
                        import json
                        with open(auto_config_path, 'r') as f:
                            config = json.load(f)
                        logger.info(f"📋 Auto-found config: {auto_config_path}")
                    else:
                        logger.warning("⚠️ No config.json found - will need to copy from original model")
            else:
                logger.info("📋 Skipping config.json - using tensor-based detection")
                # Use tensor-based config detection when config.json is skipped
                from .vibevoice_model_detection import detect_vibevoice_config_from_gguf_state_dict, vibevoice_key_map_replace, VIBEVOICE_GGUF_KEY_MAP
                
                # Apply key mapping first
                mapped_state_dict = vibevoice_key_map_replace(state_dict, VIBEVOICE_GGUF_KEY_MAP)
                
                # Detect config from mapped tensors
                model_config = detect_vibevoice_config_from_gguf_state_dict(mapped_state_dict)
                if model_config:
                    config = model_config.config
                    logger.info("📋 Generated config from GGUF tensor analysis")
                else:
                    logger.warning("⚠️ Failed to detect config from GGUF tensors")
            
            return state_dict, config
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model: {e}")
            raise RuntimeError(f"GGUF loading failed: {e}")
    
    def is_gguf_model(self, model_path: str) -> bool:
        """Check if a model path contains a GGUF model"""
        if os.path.isfile(model_path) and model_path.endswith('.gguf'):
            return True
        
        if os.path.isdir(model_path):
            # Check for model.gguf file in directory
            gguf_file = os.path.join(model_path, "model.gguf")
            return os.path.exists(gguf_file)
        
        return False
    
    def create_huggingface_compatible_model(self, state_dict: Dict[str, torch.Tensor], 
                                          config: Dict[str, Any], 
                                          model_dir: str) -> str:
        """
        Create a HuggingFace-compatible model directory from GGUF state dict.
        
        Args:
            state_dict: PyTorch state dict from GGUF
            config: Model configuration
            model_dir: Directory to save the converted model
            
        Returns:
            Path to the created model directory
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save state dict as safetensors (VibeVoice's preferred format)
        try:
            from safetensors.torch import save_file
            safetensors_path = os.path.join(model_dir, "model.safetensors")
            save_file(state_dict, safetensors_path)
            logger.info(f"💾 Saved converted model: {safetensors_path}")
        except ImportError:
            # Fallback to PyTorch format
            torch_path = os.path.join(model_dir, "pytorch_model.bin")
            torch.save(state_dict, torch_path)
            logger.info(f"💾 Saved converted model (PyTorch): {torch_path}")
        
        # Save config.json
        if config:
            import json
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"📋 Saved config: {config_path}")
        
        return model_dir


# Global instance
gguf_loader = VibeVoiceGGUFLoader()