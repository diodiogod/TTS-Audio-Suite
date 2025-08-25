#!/usr/bin/env python3
"""
Test script to verify TTS models are properly integrated with ComfyUI model management
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    import comfy.model_management as model_management
    print("✅ ComfyUI model_management imported successfully")
    
    # Check current loaded models
    if hasattr(model_management, 'current_loaded_models'):
        loaded_models = model_management.current_loaded_models
        print(f"📊 Currently loaded models: {len(loaded_models)}")
        
        for i, model in enumerate(loaded_models):
            try:
                model_info = f"Model {i}: {type(model).__name__}"
                if hasattr(model, 'model') and model.model:
                    actual_model = model.model
                    model_info += f" -> {type(actual_model).__name__}"
                    if hasattr(actual_model, 'model_info'):
                        info = actual_model.model_info
                        model_info += f" (engine={info.engine}, type={info.model_type})"
                print(f"  {model_info}")
            except Exception as e:
                print(f"  Model {i}: Error getting info - {e}")
    
    # Test unload_all_models function
    print("\n🧪 Testing unload_all_models()...")
    try:
        model_management.unload_all_models()
        print("✅ unload_all_models() completed successfully")
        
        # Check models after unload
        loaded_models = model_management.current_loaded_models
        print(f"📊 Models after unload: {len(loaded_models)}")
        
    except Exception as e:
        print(f"❌ unload_all_models() failed: {e}")
        
except ImportError:
    print("❌ ComfyUI not available - run this from ComfyUI environment")
except Exception as e:
    print(f"❌ Error: {e}")