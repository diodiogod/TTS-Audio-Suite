#!/usr/bin/env python3
"""
Minimal GGUF test script - Run this in ComfyUI environment
Usage: cd to ComfyUI directory and run: python custom_nodes/ComfyUI_TTS_Audio_Suite/test_gguf_minimal.py
"""

import sys
import os

# Add project to path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_gguf_loading():
    """Test GGUF loading without full ComfyUI"""
    print("🧪 Testing GGUF loading...")
    
    try:
        # Test basic imports
        print("📦 Testing imports...")
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import gguf
        print(f"✅ GGUF package available")
        
        # Test VibeVoice imports
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        print(f"✅ VibeVoice imports successful")
        
        # Test our GGUF loader
        from engines.vibevoice_engine.gguf_loader import VibeVoiceGGUFLoader
        print(f"✅ GGUF loader import successful")
        
        # Test GGUF ops
        from engines.vibevoice_engine.gguf_vibevoice_ops import create_vibevoice_with_gguf_ops
        print(f"✅ GGUF ops import successful")
        
        # Test loading GGUF tensors only (no model creation)
        model_path = r"J:\stablediffusion1111s2\Data\Packages\ComfyUIPy129\ComfyUI\models\TTS\vibevoice\vibevoice-7B-Q8-gguf"
        
        if not os.path.exists(model_path):
            print(f"❌ Model path not found: {model_path}")
            return False
        
        print(f"📁 Testing GGUF tensor loading from: {model_path}")
        
        loader = VibeVoiceGGUFLoader()
        gguf_file = os.path.join(model_path, "model.gguf")
        config_file = os.path.join(model_path, "config.json")
        
        if not os.path.exists(gguf_file):
            print(f"❌ GGUF file not found: {gguf_file}")
            return False
        
        # Load tensors only
        print("🔄 Loading GGUF tensors...")
        state_dict, config = loader.load_gguf_model(gguf_file, config_file, keep_quantized=True)
        
        print(f"✅ Loaded {len(state_dict)} tensors")
        print(f"📊 Sample tensor keys: {list(state_dict.keys())[:5]}")
        
        # Test config
        if config:
            print(f"✅ Config loaded with {len(config)} keys")
            model_config = VibeVoiceConfig.from_dict(config)
            print(f"✅ VibeVoice config created")
        else:
            print(f"❌ No config loaded")
            return False
        
        # Test custom Linear creation (without full model)
        print("🧪 Testing custom Linear layer...")
        from engines.vibevoice_engine.gguf_vibevoice_ops import VibeVoiceGGUFOps
        
        linear = VibeVoiceGGUFOps.Linear(512, 1024, bias=True)
        print(f"✅ Custom Linear layer created: {linear}")
        
        # Test state dict loading simulation
        print("🧪 Testing state dict loading simulation...")
        fake_state_dict = {
            'test.weight': list(state_dict.values())[0],  # Use first tensor
        }
        
        # This should work without errors
        linear._load_from_state_dict(fake_state_dict, 'test.', None, True, [], [], [])
        print(f"✅ State dict loading works")
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test actual model creation (this will be slow)"""
    print("⚠️ Testing full model creation (this will be slow)...")
    
    try:
        model_path = r"J:\stablediffusion1111s2\Data\Packages\ComfyUIPy129\ComfyUI\models\TTS\vibevoice\vibevoice-7B-Q8-gguf"
        device = torch.device('cpu')  # Use CPU to be safe
        
        from engines.vibevoice_engine.gguf_vibevoice_ops import create_vibevoice_with_gguf_ops
        
        print("🔄 Creating model with GGUF ops...")
        model, success = create_vibevoice_with_gguf_ops(model_path, device)
        
        if success:
            print("✅ Model creation successful!")
            return True
        else:
            print("❌ Model creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting minimal GGUF tests...")
    
    # Test basic functionality first
    if test_gguf_loading():
        print("\n" + "="*50)
        print("Basic tests passed. Testing model creation...")
        test_model_creation()
    else:
        print("Basic tests failed. Skipping model creation.")