#!/usr/bin/env "J:/stablediffusion1111s2/Data/Packages/ComfyUIPy129/test_env_error/Scripts/python.exe"
"""
Simple test to verify config detection fix
"""

import sys
import os
import time

# Add paths for Windows Python (use Windows format)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.join(current_dir, 'engines', 'vibevoice_engine')
sys.path.insert(0, current_dir)
sys.path.insert(0, engine_dir)

print(f"[DEBUG] Current dir: {current_dir}")
print(f"[DEBUG] Engine dir: {engine_dir}")
print(f"[DEBUG] Engine dir exists: {os.path.exists(engine_dir)}")

def test_config_detection():
    """Test that config detection works and uses actual tensor shapes"""
    try:
        # Direct import since we added the path
        import gguf_loader
        import vibevoice_model_detection
        
        print("[TEST] Testing GGUF config detection...")
        
        gguf_path = "J:/stablediffusion1111s2/Data/Packages/ComfyUIPy129/ComfyUI/models/TTS/vibevoice/vibevoice-7B-Q8-gguf/model.gguf"
        
        if not os.path.exists(gguf_path):
            print(f"[ERROR] GGUF file not found: {gguf_path}")
            return False
        
        # Load GGUF using ComfyUI-GGUF loader approach
        print("[INFO] Loading GGUF using ComfyUI-GGUF approach...")
        
        # Import ComfyUI-GGUF loader
        gguf_ref_dir = os.path.join(current_dir, 'IgnoredForGitHubDocs', 'For_reference', 'ComfyUI-GGUF')
        sys.path.insert(0, gguf_ref_dir)
        print(f"[DEBUG] GGUF ref dir: {gguf_ref_dir}")
        print(f"[DEBUG] GGUF ref exists: {os.path.exists(gguf_ref_dir)}")
        from loader import gguf_sd_loader
        
        # Load state dict
        start_time = time.time()
        raw_state_dict = gguf_sd_loader(gguf_path, handle_prefix=None)
        load_time = time.time() - start_time
        
        print(f"[INFO] Loaded {len(raw_state_dict)} raw tensors in {load_time:.1f}s")
        
        # Apply key mapping
        mapped_state_dict = vibevoice_model_detection.vibevoice_key_map_replace(
            raw_state_dict, 
            vibevoice_model_detection.VIBEVOICE_GGUF_KEY_MAP
        )
        
        # Count language model keys
        lm_keys = [k for k in mapped_state_dict.keys() if k.startswith('model.language_model.')]
        print(f"[INFO] Mapped {len(lm_keys)} language model tensors")
        
        # Test config detection
        print("[INFO] Testing config detection...")
        model_config = vibevoice_model_detection.detect_vibevoice_config_from_gguf_state_dict(mapped_state_dict)
        
        if model_config is None:
            print("[ERROR] Config detection failed")
            return False
        
        # Check the detected config
        lm_config = model_config.config.get('language_model', {})
        detected_hidden_size = lm_config.get('hidden_size')
        detected_vocab_size = lm_config.get('vocab_size')
        
        print(f"[INFO] Detected config:")
        print(f"   hidden_size: {detected_hidden_size}")
        print(f"   vocab_size: {detected_vocab_size}")
        
        # Verify against actual tensor shapes
        embed_key = 'model.language_model.embed_tokens.weight'
        if embed_key in mapped_state_dict:
            tensor = mapped_state_dict[embed_key]
            actual_vocab, actual_hidden = tensor.tensor_shape if hasattr(tensor, 'tensor_shape') else tensor.shape
            print(f"[INFO] Actual tensor shape: {actual_vocab} x {actual_hidden}")
            
            # Verify they match
            if int(detected_vocab_size) == int(actual_vocab) and int(detected_hidden_size) == int(actual_hidden):
                print("[SUCCESS] Config detection uses actual tensor shapes!")
                return True
            else:
                print(f"[ERROR] Mismatch: config({detected_vocab_size}, {detected_hidden_size}) vs tensor({actual_vocab}, {actual_hidden})")
                return False
        else:
            print("[ERROR] Embed tensor not found")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[START] Testing config detection fix...")
    success = test_config_detection()
    
    if success:
        print("\n[SUCCESS] Config detection fix verified!")
    else:
        print("\n[FAILED] Config detection needs more work.")