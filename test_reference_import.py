#!/usr/bin/env python3
"""
Test script to check if we can import the reference implementation
"""

import sys
import os

# Add the reference implementation to path
reference_path = os.path.join(os.path.dirname(__file__), "engines", "rvc", "reference_impl")
if reference_path not in sys.path:
    sys.path.insert(0, reference_path)

print(f"Testing import from: {reference_path}")

try:
    print("Testing config import...")
    import config
    print("✅ Config imported successfully")
    
    print("Testing vc_infer_pipeline import...")
    import vc_infer_pipeline
    print("✅ vc_infer_pipeline imported successfully")
    
    print("Testing specific functions...")
    from vc_infer_pipeline import get_vc, vc_single
    print("✅ get_vc and vc_single imported successfully")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()