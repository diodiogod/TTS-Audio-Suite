#!/usr/bin/env python3
"""
Fix relative imports in reference implementation
"""

import os
import re

def fix_imports_in_file(file_path):
    """Fix relative imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace relative imports with absolute imports
        patterns = [
            (r'from \.lib\.', 'from lib.'),
            (r'from \.config', 'from config'),
            (r'from \.pitch_extraction', 'from pitch_extraction'),
            (r'from \.', 'from '),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in: {os.path.basename(file_path)}")
            return True
        else:
            print(f"⚪ No changes needed: {os.path.basename(file_path)}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    reference_dir = "docs/RVC/Comfy-RVC-For-Reference"
    
    # Key files that need fixing for basic RVC functionality
    key_files = [
        "vc_infer_pipeline.py",
        "pitch_extraction.py", 
        "lib/model_utils.py",
        "lib/audio.py",
        "lib/utils.py"
    ]
    
    print("Fixing relative imports in key RVC reference files...")
    
    fixed_count = 0
    for file_name in key_files:
        file_path = os.path.join(reference_dir, file_name)
        if os.path.exists(file_path):
            if fix_imports_in_file(file_path):
                fixed_count += 1
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print(f"\nFixed imports in {fixed_count} files")

if __name__ == "__main__":
    main()