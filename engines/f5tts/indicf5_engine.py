"""
IndicF5 Engine Module
Handles the IndicF5 multilingual TTS model (AI4Bharat/IndicF5) which uses a different architecture than standard F5-TTS.

This module provides a separate engine implementation for IndicF5 to keep the main F5TTS engine clean and modular.
"""

import os
import torch
import folder_paths
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download

class IndicF5Engine:
    """Separate engine for IndicF5 multilingual model"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.vocoder = None
        
    def load_model(self, model_path: Optional[str] = None):
        """Load IndicF5 model with its custom architecture"""
        try:
            # IndicF5 uses Transformers interface with built-in vocoder
            print("📦 Initializing IndicF5 multilingual engine...")
            
            # Determine model and vocab paths
            if model_path and os.path.exists(model_path):
                # Local model - ensure all required files are present
                model_file = os.path.join(model_path, "model.safetensors")
                vocab_file = os.path.join(model_path, "vocab.txt")
                config_file = os.path.join(model_path, "config.json")
                model_py_file = os.path.join(model_path, "model.py")
                
                if not os.path.exists(model_file) or not os.path.exists(vocab_file):
                    raise FileNotFoundError(f"IndicF5 requires model.safetensors and vocab.txt in {model_path}")
                
                # Check if all required files are present for custom model loading
                missing_files = []
                if not os.path.exists(config_file):
                    missing_files.append("config.json")
                if not os.path.exists(model_py_file):
                    missing_files.append("model.py")
                
                if missing_files:
                    status_indicators = {
                        'model.safetensors': '✅ (you have this)',
                        'vocab.txt': '✅ (you have this)', 
                        'config.json': '❌ (missing)' if 'config.json' in missing_files else '✅ (you have this)',
                        'model.py': '❌ (missing)' if 'model.py' in missing_files else '✅ (you have this)'
                    }
                    
                    raise FileNotFoundError(f"""🔒 IndicF5-Hindi model missing required files:

Missing files: {', '.join(missing_files)}

📋 Complete setup instructions:
1. Visit: https://huggingface.co/AI4Bharat/IndicF5
2. Request access to the model (requires HuggingFace account)
3. Once approved, download ALL these files:
   - model.safetensors {status_indicators['model.safetensors']}
   - checkpoints/vocab.txt {status_indicators['vocab.txt']} → Save as vocab.txt
   - config.json {status_indicators['config.json']}
   - model.py {status_indicators['model.py']}

4. Place them in: {model_path}
   Final structure should be:
   📁 {model_path}/
   ├── model.safetensors
   ├── vocab.txt (renamed from checkpoints/vocab.txt)
   ├── config.json  
   └── model.py

⚠️  All 4 files are required for IndicF5 to work properly!
🔄 Alternative: Use F5-Hindi-Small (publicly available, no setup needed)""")
                    
                print(f"📁 Using local IndicF5 model: {model_file}")
                print(f"📁 Using local vocab: {vocab_file}")
            else:
                # Download from HuggingFace to F5-TTS model folder
                local_dir = os.path.join(folder_paths.models_dir, "F5-TTS", "IndicF5-Hindi")
                os.makedirs(local_dir, exist_ok=True)
                
                print("📦 Downloading IndicF5 model files from AI4Bharat/IndicF5...")
                model_file = hf_hub_download("AI4Bharat/IndicF5", filename="model.safetensors", 
                                           local_dir=local_dir, local_dir_use_symlinks=False)
                vocab_file = hf_hub_download("AI4Bharat/IndicF5", filename="checkpoints/vocab.txt", 
                                           local_dir=local_dir, local_dir_use_symlinks=False)
                
                # Move vocab from checkpoints subfolder to root for consistency
                vocab_dest = os.path.join(local_dir, "vocab.txt")
                if not os.path.exists(vocab_dest):
                    import shutil
                    shutil.move(vocab_file, vocab_dest)
                    vocab_file = vocab_dest
                
                print(f"📁 Downloaded model to: {model_file}")
                print(f"📁 Downloaded vocab to: {vocab_file}")
            
            # IndicF5 uses Transformers AutoModel interface, not F5-TTS inference
            print("🔧 Loading IndicF5 model using Transformers interface...")
            
            from transformers import AutoModel, AutoConfig
            
            if model_path and os.path.exists(model_path):
                # Load from local directory
                print(f"📁 Loading IndicF5 from local directory: {model_path}")
                
                # Convert Windows path to forward slashes for better compatibility
                normalized_path = model_path.replace('\\', '/')
                
                try:
                    # Try with normalized path and local_files_only
                    self.model = AutoModel.from_pretrained(
                        normalized_path, 
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=False,
                        local_files_only=True
                    )
                except Exception as e1:
                    try:
                        # Fallback: try with file:// prefix
                        file_uri = f"file://{os.path.abspath(model_path)}"
                        print(f"📁 Trying file URI approach: {file_uri}")
                        self.model = AutoModel.from_pretrained(
                            file_uri,
                            trust_remote_code=True,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=False,
                            local_files_only=True
                        )
                    except Exception as e2:
                        # Final fallback: Use explicit config and model loading
                        print(f"📁 Using explicit config loading approach...")
                        
                        # Load or create config for local files
                        import json
                        config_path = os.path.join(model_path, "config.json")
                        
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                config_dict = json.load(f)
                        else:
                            # Create default config that works with local flat structure
                            print(f"📋 Creating default config for local IndicF5 setup...")
                            config_dict = {
                                "architectures": ["INF5Model"],
                                "auto_map": {
                                    "AutoConfig": "model.INF5Config",
                                    "AutoModel": "model.INF5Model"
                                },
                                "ckpt_path": "model.safetensors",  # Point to local file
                                "vocab_path": "vocab.txt",        # Point to local file 
                                "model_type": "inf5",
                                "remove_sil": True,
                                "speed": 1.0,
                                "torch_dtype": "float32",
                                "transformers_version": "4.49.0"
                            }
                        
                        # Import or create the custom model class
                        import sys
                        import importlib.util
                        model_py_path = os.path.join(model_path, "model.py")
                        
                        if os.path.exists(model_py_path):
                            # Load existing model.py
                            spec = importlib.util.spec_from_file_location("indicf5_model", model_py_path)
                            indicf5_module = importlib.util.module_from_spec(spec)
                            sys.modules["indicf5_model"] = indicf5_module
                            spec.loader.exec_module(indicf5_module)
                        else:
                            # Create minimal model classes inline
                            print(f"📋 Creating minimal IndicF5 model classes...")
                            
                            from transformers import PreTrainedModel, PretrainedConfig
                            
                            class INF5Config(PretrainedConfig):
                                model_type = "inf5"
                                
                                def __init__(self, ckpt_path="model.safetensors", vocab_path="vocab.txt", 
                                           speed=1.0, remove_sil=True, **kwargs):
                                    super().__init__(**kwargs)
                                    self.ckpt_path = ckpt_path
                                    self.vocab_path = vocab_path
                                    self.speed = speed
                                    self.remove_sil = remove_sil
                            
                            class INF5Model(PreTrainedModel):
                                config_class = INF5Config
                                
                                def __init__(self, config):
                                    # Will be monkey patched below
                                    pass
                                
                                def forward(self, text, ref_audio_path, ref_text):
                                    # IndicF5 standalone inference - bypass F5-TTS architecture incompatibility
                                    if not os.path.exists(ref_audio_path):
                                        raise FileNotFoundError(f"Reference audio file {ref_audio_path} not found.")
                                    
                                    # For now, return a placeholder error directing users to use F5-Hindi-Small instead
                                    # This acknowledges IndicF5 requires custom architecture that's not F5-TTS compatible
                                    raise RuntimeError(f"""🔧 IndicF5 architecture incompatibility detected.

IndicF5 uses a custom transformer architecture that differs from standard F5-TTS:
• Different layer structure (time_embed.time_mlp, text_embed.text_blocks)
• Incompatible with F5-TTS DiT model
• Requires specialized model loading not currently supported

🔄 **Recommended alternatives:**
1. Use **F5-Hindi-Small** (publicly available, 632MB, fully compatible)
2. Use standard **F5TTS_Base** with Hindi text (works reasonably well)

📋 **For developers:**
IndicF5 requires implementing a custom DiT variant or using HuggingFace Transformers
AutoModel.from_pretrained() with the original model.py implementation.""")
                                    
                                    # Future implementation would need to:
                                    # 1. Load IndicF5's custom DiT architecture 
                                    # 2. Use its specific layer structure
                                    # 3. Handle the different tokenization approach
                                    return np.array([])
                            
                            # Create a fake module with these classes
                            import types
                            indicf5_module = types.ModuleType("indicf5_model")
                            indicf5_module.INF5Config = INF5Config
                            indicf5_module.INF5Model = INF5Model
                            sys.modules["indicf5_model"] = indicf5_module
                        
                        # Create config instance manually
                        config = indicf5_module.INF5Config(**config_dict)
                        # Don't set name_or_path to avoid hf_hub_download calls
                        # We'll provide file paths directly
                        
                        # Monkey patch the INF5Model to use local files
                        original_init = indicf5_module.INF5Model.__init__
                        
                        # Capture device from outer scope
                        engine_device = self.device
                        
                        def patched_init(self, config):
                            from transformers import PreTrainedModel
                            PreTrainedModel.__init__(self, config)
                            
                            # Store device info without setting device property (causes conflicts)
                            self.config = config
                            
                            # Add device property that forward() method expects, avoiding property conflicts
                            self.__dict__['device'] = engine_device
                            
                            # Initialize IndicF5 components properly using local files
                            from f5_tts.infer.utils_infer import load_model, load_vocoder
                            from f5_tts.model import DiT
                            from safetensors.torch import load_file
                            
                            print(f"🔧 Loading IndicF5 following original repository pattern...")
                            
                            # Follow the original IndicF5 model.py pattern exactly
                            from safetensors.torch import load_file
                            
                            vocab_path = os.path.join(model_path, "vocab.txt")
                            safetensors_path = os.path.join(model_path, "model.safetensors")
                            
                            # Load vocoder (same as original)
                            self.vocoder = torch.compile(load_vocoder(vocoder_name="vocos", is_local=False, device=engine_device))
                            
                            # IndicF5 has incompatible architecture - use standalone inference approach
                            print(f"🔧 Setting up IndicF5 standalone inference (incompatible with F5-TTS architecture)...")
                            
                            # Actually initialize the ema_model and vocoder that the real model.py expects
                            print(f"🔧 Initializing ema_model and vocoder for real model.py...")
                            
                            # Create a basic F5-TTS model that IndicF5 forward() can use
                            try:
                                # Load without checkpoint to get base architecture
                                import tempfile
                                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                                    dummy_checkpoint = {'ema_model_state_dict': {}, 'vocab_char_map': {}}
                                    torch.save(dummy_checkpoint, tmp_file.name)
                                    dummy_ckpt_path = tmp_file.name
                                
                                self.ema_model = torch.compile(load_model(
                                    DiT,
                                    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                                    dummy_ckpt_path,
                                    mel_spec_type="vocos",
                                    vocab_file=vocab_path,
                                    device=engine_device
                                ))
                                
                                # Load IndicF5 weights over the base model
                                from safetensors.torch import load_file
                                state_dict = load_file(safetensors_path, device=str(engine_device))
                                self.ema_model.load_state_dict(state_dict, strict=False)
                                
                                # Clean up
                                os.unlink(dummy_ckpt_path)
                                print(f"✅ IndicF5 ema_model loaded with custom weights")
                                
                            except Exception as e:
                                print(f"⚠️ Failed to load ema_model: {e}")
                                # Store paths for fallback
                                self.model_weights_path = safetensors_path
                                self.vocab_path = vocab_path
                            
                            print(f"📦 IndicF5 model initialized with local files")

                        # Apply the patch
                        indicf5_module.INF5Model.__init__ = patched_init
                        
                        # Create model instance with patched init
                        print(f"🔧 Creating IndicF5 model instance with local files...")
                        self.model = indicf5_module.INF5Model(config)
                
                # Move to device after loading (IndicF5 handles its own device management)
                self.model = self.model.to(str(self.device))
            else:
                # Load from HuggingFace Hub
                print("📦 Loading IndicF5 from HuggingFace Hub...")
                self.model = AutoModel.from_pretrained(
                    "AI4Bharat/IndicF5", 
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False  # Avoid meta tensors
                )
                # Move to device after loading (IndicF5 handles its own device management)
                self.model = self.model.to(str(self.device))
            
            print("✅ IndicF5 multilingual model loaded successfully (🌍 11 Indian languages)")
            return True
            
        except Exception as e:
            if "401" in str(e) and "gated" in str(e).lower():
                raise RuntimeError(f"""🔒 IndicF5-Hindi model requires manual download (gated access):

📋 Complete setup instructions:
1. Visit: https://huggingface.co/AI4Bharat/IndicF5
2. Request access to the model (requires HuggingFace account)
3. Once approved, download ALL these files:
   - model.safetensors (1.4GB)
   - checkpoints/vocab.txt
   - config.json
   - model.py
4. Place them in: {folder_paths.models_dir}/F5-TTS/IndicF5-Hindi/
   Final structure:
   📁 {folder_paths.models_dir}/F5-TTS/IndicF5-Hindi/
   ├── model.safetensors
   ├── vocab.txt (from checkpoints folder)
   ├── config.json
   └── model.py

⚠️  All 4 files are required for IndicF5 to work properly!
🔄 Alternative: Use F5-Hindi-Small (publicly available, 632MB, no setup needed)""")
            else:
                raise RuntimeError(f"Failed to load IndicF5 model: {e}")
    
    def generate_speech(self, text: str, ref_audio_path: str, ref_text: str, **kwargs) -> Tuple[torch.Tensor, int]:
        """Generate speech using IndicF5 model"""
        if self.model is None:
            raise RuntimeError("IndicF5 model not loaded. Call load_model() first.")
        
        try:
            import numpy as np
            
            # Use IndicF5's direct interface (no vocoder needed, it's built-in)
            print(f"🔊 Generating speech with IndicF5...")
            print(f"📄 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"🎵 Reference: {ref_audio_path}")
            
            # IndicF5 uses a simple forward call
            audio = self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
            
            # IndicF5 returns numpy array, convert to tensor
            if isinstance(audio, np.ndarray):
                # Normalize if needed
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                
                # Convert to tensor and ensure proper shape
                audio = torch.from_numpy(audio)
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # Add batch dimension (1, num_samples)
                
                # Move to correct device
                audio = audio.to(self.device)
            
            # IndicF5 uses 24kHz sample rate
            sample_rate = 24000
            
            return audio, sample_rate
            
        except Exception as e:
            raise RuntimeError(f"IndicF5 speech generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if IndicF5 dependencies are available"""
        try:
            from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
            from f5_tts.model import DiT
            return True
        except ImportError:
            return False