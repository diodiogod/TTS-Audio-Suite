#!/usr/bin/env "J:/stablediffusion1111s2/Data/Packages/ComfyUIPy129/python.exe"
"""
Test script to verify GGUF config detection fix
"""

import sys
import os
import time
import torch

# Add current directory to path
sys.path.insert(0, '/home/linux/ComfyUI_TTS_Audio_Suite')
sys.path.insert(0, '/home/linux/ComfyUI_TTS_Audio_Suite/engines/vibevoice_engine')

def test_gguf_config_detection():
    """Test that config detection uses GGUF tensor shapes"""
    # Add engine path to sys.path temporarily for imports  
    engine_path = '/home/linux/ComfyUI_TTS_Audio_Suite/engines/vibevoice_engine'
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    
    from gguf_loader import VibeVoiceGGUFLoader
    
    print("[TEST] Testing GGUF config detection fix...")
    
    # Test GGUF file path (Windows path for Windows Python)
    gguf_path = "J:/stablediffusion1111s2/Data/Packages/ComfyUIPy129/ComfyUI/models/TTS/vibevoice/vibevoice-7B-Q8-gguf/model.gguf"
    
    if not os.path.exists(gguf_path):
        print(f"[ERROR] GGUF file not found: {gguf_path}")
        return False
    
    try:
        loader = VibeVoiceGGUFLoader()
        start_time = time.time()
        
        # Load just the state dict and config
        state_dict, config = loader.load_gguf_model(
            gguf_path,
            config_path=None,  # Don't use config.json
            keep_quantized=True
        )
        
        load_time = time.time() - start_time
        print(f"[INFO] Loaded {len(state_dict)} tensors in {load_time:.1f}s")
        
        # Check config values
        lm_config = config.get('language_model', {})
        hidden_size = lm_config.get('hidden_size')
        vocab_size = lm_config.get('vocab_size')
        
        print(f"[INFO] Config from GGUF tensors:")
        print(f"   hidden_size: {hidden_size}")
        print(f"   vocab_size: {vocab_size}")
        
        # Check actual tensor shapes
        embed_key = 'model.language_model.embed_tokens.weight'
        if embed_key in state_dict:
            tensor = state_dict[embed_key]
            actual_vocab, actual_hidden = tensor.tensor_shape if hasattr(tensor, 'tensor_shape') else tensor.shape
            print(f"[INFO] Actual embed tensor shape: {actual_vocab} x {actual_hidden}")
            
            # Verify they match
            if int(vocab_size) == int(actual_vocab) and int(hidden_size) == int(actual_hidden):
                print(f"[SUCCESS] Config matches tensor shapes!")
                return True
            else:
                print(f"[ERROR] Config mismatch: config({vocab_size}, {hidden_size}) vs tensor({actual_vocab}, {actual_hidden})")
                return False
        else:
            print(f"[ERROR] Embed tensor not found in state dict")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lazy_model_creation():
    """Test that lazy model creates without tensor shape mismatches"""
    # Add engine path to sys.path temporarily for imports  
    engine_path = '/home/linux/ComfyUI_TTS_Audio_Suite/engines/vibevoice_engine'
    if engine_path not in sys.path:
        sys.path.insert(0, engine_path)
    
    from vibevoice_lazy_model import create_lazy_vibevoice_from_gguf
    
    print("\n[TEST] Testing lazy model creation...")
    
    model_path = "J:/stablediffusion1111s2/Data/Packages/ComfyUIPy129/ComfyUI/models/TTS/vibevoice/vibevoice-7B-Q8-gguf"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model path not found: {model_path}")
        return False
    
    try:
        start_time = time.time()
        model, is_gguf = create_lazy_vibevoice_from_gguf(model_path, torch.device('cpu'))
        creation_time = time.time() - start_time
        
        print(f"[INFO] Lazy model created in {creation_time:.1f}s")
        print(f"[INFO] Is GGUF: {is_gguf}")
        
        # Test parameters() method
        params = list(model.parameters())
        print(f"[INFO] Model parameters available: {len(params)}")
        
        if len(params) > 0:
            device = next(model.parameters()).device
            print(f"[INFO] Model device: {device}")
        
        print(f"[SUCCESS] Lazy model creation successful!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Lazy model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[START] Testing GGUF fixes...")
    
    success1 = test_gguf_config_detection()
    success2 = test_lazy_model_creation()
    
    if success1 and success2:
        print(f"\n[SUCCESS] All tests passed! GGUF fix is working correctly.")
    else:
        print(f"\n[FAILED] Some tests failed. Check the output above.")