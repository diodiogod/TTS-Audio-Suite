"""
VibeVoice GGUF Detection and Loading
Replicates ComfyUI's model detection and loading approach for VibeVoice
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class VibeVoiceGGUFConfig:
    """
    Minimal config detected from GGUF state dict, similar to ComfyUI's model configs
    """
    def __init__(self, detected_config: dict, custom_operations=None):
        self.config = detected_config
        self.custom_operations = custom_operations
        self.supported_inference_dtypes = [torch.float16, torch.float32, torch.bfloat16]
    
    def get_model(self, state_dict: Dict[str, Any], prefix: str = ""):
        """
        Create VibeVoice model from detected config and state dict
        This is the equivalent of ComfyUI's model_config.get_model()
        """
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        
        print(f"🔧 Creating VibeVoice model from detected GGUF config...")
        
        # Create config from our detected parameters
        vibevoice_config = VibeVoiceConfig.from_dict(self.config)
        
        # Replace nn.Linear with custom ops if provided
        if self.custom_operations and hasattr(self.custom_operations, 'Linear'):
            original_linear = nn.Linear
            nn.Linear = self.custom_operations.Linear
            
            try:
                # Create model with custom ops
                model = VibeVoiceForConditionalGenerationInference(vibevoice_config)
                return VibeVoiceGGUFModel(model, state_dict)
            finally:
                nn.Linear = original_linear
        else:
            # Create model normally
            model = VibeVoiceForConditionalGenerationInference(vibevoice_config)
            return VibeVoiceGGUFModel(model, state_dict)


class VibeVoiceGGUFModel:
    """
    Wrapper for VibeVoice model that handles GGUF weight loading
    Similar to ComfyUI's model wrappers
    """
    def __init__(self, model, state_dict):
        self.model = model
        self.state_dict = state_dict
    
    def load_model_weights(self, state_dict: Dict[str, Any], prefix: str = ""):
        """
        Load GGUF weights into the model
        This is the equivalent of ComfyUI's model.load_model_weights()
        """
        print(f"📥 Loading GGUF weights into VibeVoice model...")
        
        # Load state dict with custom handling for GGUF tensors
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        print(f"📊 GGUF weights loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"   Missing keys: {missing[:5]}...")
        if len(unexpected) > 0:
            print(f"   Unexpected keys: {unexpected[:5]}...")
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to eval mode"""
        self.model.eval()
        return self
    
    def __call__(self, *args, **kwargs):
        """Forward pass"""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate method"""
        return self.model.generate(*args, **kwargs)


def vibevoice_config_from_gguf_state_dict(state_dict: Dict[str, Any]) -> Optional[VibeVoiceGGUFConfig]:
    """
    Detect VibeVoice configuration from GGUF state dict
    This is the equivalent of ComfyUI's model_config_from_unet()
    """
    print(f"🔍 Detecting VibeVoice architecture from GGUF state dict...")
    
    keys = set(state_dict.keys())
    
    # Check if this looks like a VibeVoice model
    vibevoice_indicators = [
        'model.acoustic_tokenizer',
        'model.prediction_head', 
        'model.language_model',
        'lm_head.weight'
    ]
    
    has_vibevoice_components = sum(1 for indicator in vibevoice_indicators 
                                   if any(indicator in key for key in keys))
    
    if has_vibevoice_components < 3:
        print(f"❌ Does not appear to be a VibeVoice model (found {has_vibevoice_components}/4 components)")
        return None
    
    print(f"✅ Detected VibeVoice model with {has_vibevoice_components}/4 components")
    
    # Extract configuration from tensor shapes
    detected_config = extract_config_from_tensors(state_dict, keys)
    
    if detected_config:
        print(f"✅ Successfully detected VibeVoice configuration")
        return VibeVoiceGGUFConfig(detected_config)
    else:
        print(f"❌ Failed to extract configuration from tensors")
        return None


def extract_config_from_tensors(state_dict: Dict[str, Any], keys: set) -> Optional[dict]:
    """
    Extract VibeVoice configuration by analyzing tensor shapes
    """
    config = {
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "model_type": "vibevoice"
    }
    
    try:
        # Extract language model config
        lm_config = {}
        
        # Get vocab size and hidden size from embedding layer
        embed_keys = [k for k in keys if 'language_model.embed_tokens.weight' in k]
        if embed_keys:
            embed_tensor = state_dict[embed_keys[0]]
            vocab_size, hidden_size = get_tensor_shape(embed_tensor)
            lm_config['vocab_size'] = int(vocab_size)
            lm_config['hidden_size'] = int(hidden_size)
            print(f"📊 Detected LM: vocab_size={vocab_size}, hidden_size={hidden_size}")
        
        # Count transformer layers
        layer_keys = [k for k in keys if 'language_model.layers.' in k and '.self_attn.q_proj.weight' in k]
        if layer_keys:
            layer_numbers = []
            for key in layer_keys:
                try:
                    # Extract layer number from key like "model.language_model.layers.31.self_attn.q_proj.weight"
                    parts = key.split('.')
                    layer_idx = next(int(part) for part in parts if part.isdigit())
                    layer_numbers.append(layer_idx)
                except:
                    continue
            
            if layer_numbers:
                num_layers = max(layer_numbers) + 1
                lm_config['num_hidden_layers'] = int(num_layers)
                print(f"📊 Detected {num_layers} transformer layers")
        
        # Detect attention heads
        q_proj_keys = [k for k in keys if 'self_attn.q_proj.weight' in k]
        if q_proj_keys and 'hidden_size' in lm_config:
            q_tensor = state_dict[q_proj_keys[0]]
            q_out_dim, q_in_dim = get_tensor_shape(q_tensor)
            
            # Calculate number of attention heads
            hidden_size = lm_config['hidden_size']
            if q_out_dim % hidden_size == 0:
                num_attention_heads = q_out_dim // (hidden_size // 32)  # Assume head_dim=32 as common
                lm_config['num_attention_heads'] = min(int(num_attention_heads), 64)  # Cap at reasonable value
                print(f"📊 Detected ~{lm_config['num_attention_heads']} attention heads")
        
        # Set reasonable defaults for missing values
        lm_config.setdefault('intermediate_size', lm_config.get('hidden_size', 3584) * 4)
        lm_config.setdefault('num_attention_heads', 32)
        lm_config.setdefault('num_key_value_heads', 32)
        lm_config.setdefault('max_position_embeddings', 32768)
        lm_config.setdefault('rms_norm_eps', 1e-6)
        lm_config.setdefault('rope_theta', 1000000.0)
        lm_config.setdefault('attention_dropout', 0.0)
        lm_config.setdefault('model_type', 'qwen2')
        
        config['language_model'] = lm_config
        
        # Extract acoustic tokenizer config
        acoustic_config = {}
        acoustic_keys = [k for k in keys if 'acoustic_tokenizer' in k]
        if acoustic_keys:
            # Set reasonable defaults
            acoustic_config = {
                "num_acoustic_tokens": 64,
                "acoustic_vocab_size": 4096,
                "sampling_rate": 24000
            }
        config['acoustic_tokenizer'] = acoustic_config
        
        # Extract prediction head config  
        pred_head_config = {}
        pred_keys = [k for k in keys if 'prediction_head' in k]
        if pred_keys:
            pred_head_config = {
                "hidden_size": lm_config.get('hidden_size', 3584),
                "num_layers": 4
            }
        config['prediction_head'] = pred_head_config
        
        # Add other required config fields
        config.update({
            'torch_dtype': 'float16',
            'use_cache': True,
            'pad_token_id': 151643,
            'bos_token_id': 151643,
            'eos_token_id': 151645
        })
        
        return config
        
    except Exception as e:
        print(f"❌ Failed to extract config from tensors: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_tensor_shape(tensor) -> tuple:
    """Get shape from tensor, handling GGUF tensors"""
    if hasattr(tensor, 'tensor_shape'):
        return tensor.tensor_shape
    elif hasattr(tensor, 'shape'):
        return tensor.shape
    else:
        return tensor.size()


def load_vibevoice_from_gguf_state_dict(state_dict: Dict[str, Any], model_options: dict = {}) -> Optional[Any]:
    """
    Load VibeVoice from GGUF state dict using ComfyUI's approach
    This is the equivalent of ComfyUI's load_diffusion_model_state_dict()
    """
    print(f"🚀 Loading VibeVoice from GGUF state dict (ComfyUI approach)...")
    
    # Detect model config from state dict
    model_config = vibevoice_config_from_gguf_state_dict(state_dict)
    
    if model_config is None:
        print(f"❌ Could not detect VibeVoice configuration from state dict")
        return None
    
    # Set custom operations
    custom_ops = model_options.get("custom_operations", None)
    model_config.custom_operations = custom_ops
    
    # Create model using detected config
    model = model_config.get_model(state_dict, "")
    
    # Load weights
    model.load_model_weights(state_dict, "")
    
    print(f"✅ VibeVoice model loaded from GGUF state dict")
    return model