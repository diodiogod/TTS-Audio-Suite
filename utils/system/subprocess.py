#!/usr/bin/env python3
"""
ChatterBox TTS Subprocess Wrapper

This script runs ChatterBox TTS generation in an isolated subprocess to prevent
CUDA crashes from affecting the main ComfyUI process.

Usage:
    python chatterbox_subprocess.py --text "Hello world" --reference_audio "path/to/ref.wav" --output "output.wav"
"""

import sys
import os
import argparse
import json
import traceback
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='ChatterBox TTS Subprocess')
    parser.add_argument('--text', required=True, help='Text to synthesize')
    parser.add_argument('--reference_audio', required=True, help='Path to reference audio')
    parser.add_argument('--output', required=True, help='Output audio file path')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--exaggeration', type=float, default=0.5, help='Exaggeration factor')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature')
    parser.add_argument('--cfg_weight', type=float, default=0.5, help='CFG weight')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        # Import ChatterBox TTS modules
        from engines.chatterbox.tts import ChatterboxTTS
        import torchaudio
        
        print(f"🔄 Subprocess: Loading ChatterBox TTS on {args.device}...")
        
        # Initialize ChatterBox TTS
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        chatterbox = ChatterboxTTS.from_pretrained(device=device)
        
        print(f"🔄 Subprocess: Loading reference audio from {args.reference_audio}...")
        
        # Load reference audio
        ref_audio, sample_rate = torchaudio.load(args.reference_audio)
        
        # Ensure mono audio
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != chatterbox.sr:
            resampler = torchaudio.transforms.Resample(sample_rate, chatterbox.sr)
            ref_audio = resampler(ref_audio)
        
        print(f"🔄 Subprocess: Generating speech for text: '{args.text[:50]}...'")
        
        # Set seed for reproducibility
        if args.seed != 0:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
        
        # Generate audio
        generated_audio = chatterbox.generate(
            text=args.text,
            reference_audio=ref_audio,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight
        )
        
        print(f"🔄 Subprocess: Saving audio to {args.output}...")
        
        # Save output audio
        torchaudio.save(args.output, generated_audio.cpu(), chatterbox.sr)
        
        # Return success info
        duration = generated_audio.size(-1) / chatterbox.sr
        result = {
            'success': True,
            'output_path': args.output,
            'duration': duration,
            'sample_rate': chatterbox.sr,
            'audio_shape': list(generated_audio.shape)
        }
        
        print(f"✅ Subprocess: Generation completed successfully ({duration:.2f}s)")
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"❌ Subprocess: Generation failed: {e}")
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()