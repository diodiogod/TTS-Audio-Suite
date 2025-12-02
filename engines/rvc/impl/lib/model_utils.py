
import hashlib
import torch.nn.functional as F
import librosa
import torch
import os

from .infer_pack.loaders import HubertModelWithFinalProj

def get_hash(model_path):
    try:
        with open(model_path, 'rb') as f:
            f.seek(- 10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except:
        model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

    return model_hash

def load_hubert(model_path: str, config):
    try:
        print(f"🔧 Loading HuBERT model: {model_path}")
        if model_path.endswith(".safetensors"):
            # Check if this safetensors file needs migration to subdirectory
            from engines.rvc.hubert_models import HUBERT_MODELS

            model_filename = os.path.basename(model_path)
            model_dir_path = os.path.dirname(model_path)

            # Check if this is a transformers model that should be in a subdirectory
            for key, info in HUBERT_MODELS.items():
                if info.get('is_transformers') and info.get('filename') == model_filename and info.get('model_dir'):
                    # Check if currently in flat directory (not already in subdirectory)
                    if os.path.basename(model_dir_path) != info['model_dir']:
                        target_subdir = os.path.join(model_dir_path, info['model_dir'])
                        target_path = os.path.join(target_subdir, model_filename)

                        # Migrate to subdirectory
                        print(f"🔄 Migrating {model_filename} to subdirectory: {info['model_dir']}")
                        os.makedirs(target_subdir, exist_ok=True)

                        try:
                            import shutil
                            shutil.move(model_path, target_path)
                            model_path = target_path  # Update path to new location
                            print(f"✅ Migrated to: {target_path}")

                            # Download config if missing
                            from engines.rvc.hubert_downloader import _download_transformers_config
                            config_path = os.path.join(target_subdir, 'config.json')
                            if not os.path.exists(config_path) and info.get('repo_id'):
                                print(f"📥 Downloading config.json for {key}...")
                                _download_transformers_config(info['repo_id'], target_subdir)
                        except Exception as migrate_error:
                            print(f"⚠️ Migration failed: {migrate_error}")
                    break

            try:
                model = HubertModelWithFinalProj.from_safetensors(model_path, device=config.device)
                print(f"✅ HuBERT safetensors model loaded successfully")
                return model
            except Exception as e:
                print(f"❌ HuBERT safetensors loading error: {e}")
                print(f"🔧 Error type: {type(e).__name__}")
                raise
        else:
            # Try loading .pt file directly first (safer approach)
            print(f"🔧 Attempting direct .pt loading: {model_path}")
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Try to create HuBERT model directly from .pt checkpoint
                # This will work if the checkpoint contains the full model with config
                if hasattr(checkpoint, 'eval'):
                    # The checkpoint itself is a model
                    model = checkpoint.to(config.device)
                    model.eval()
                    print(f"✅ Direct .pt model loading successful")
                    return model

            except Exception as direct_error:
                print(f"⚠️ Direct .pt loading failed: {direct_error}")
                print(f"🔄 Falling back to safetensors conversion")

            # Fallback: Convert .pt file to .safetensors format for compatibility
            print(f"🔄 Converting .pt Hubert model to safetensors format: {model_path}")

            # Determine if this is a transformers model that needs a subdirectory
            # Import here to avoid circular dependency
            from engines.rvc.hubert_models import HUBERT_MODELS

            # Try to identify which model this is by filename
            model_filename = os.path.basename(model_path)
            target_subdir = None
            model_key = None

            for key, info in HUBERT_MODELS.items():
                # Check if this filename matches a transformers model
                if info.get('is_transformers') and info.get('filename'):
                    # Match both .pt and .safetensors versions
                    expected_pt = info['filename'].replace('.safetensors', '.pt')
                    expected_safetensors = info['filename']
                    if model_filename in [expected_pt, expected_safetensors, info['filename']]:
                        target_subdir = info.get('model_dir')
                        model_key = key
                        break

            # Generate safetensors path - use subdirectory if this is a transformers model
            if target_subdir:
                # Save to subdirectory
                base_dir = os.path.dirname(model_path)
                subdir_path = os.path.join(base_dir, target_subdir)
                os.makedirs(subdir_path, exist_ok=True)
                safetensors_filename = model_filename.replace(".pt", ".safetensors")
                safetensors_path = os.path.join(subdir_path, safetensors_filename)
                print(f"🔧 Transformers model detected ({model_key}), saving to subdirectory: {target_subdir}")

                # Check if safetensors already exists in flat directory and needs migration
                flat_safetensors = model_path.replace(".pt", ".safetensors")
                if os.path.exists(flat_safetensors) and not os.path.exists(safetensors_path):
                    print(f"🔄 Migrating existing safetensors to subdirectory...")
                    try:
                        import shutil
                        shutil.move(flat_safetensors, safetensors_path)
                        print(f"✅ Migrated: {os.path.basename(flat_safetensors)} -> {target_subdir}/")

                        # Also download config.json if missing
                        try:
                            from engines.rvc.hubert_downloader import _download_transformers_config
                            info = HUBERT_MODELS.get(model_key)
                            if info and info.get('repo_id'):
                                config_dir = os.path.dirname(safetensors_path)
                                config_path = os.path.join(config_dir, 'config.json')
                                if not os.path.exists(config_path):
                                    print(f"📥 Downloading config.json for {model_key}...")
                                    _download_transformers_config(info['repo_id'], config_dir)
                        except Exception as config_error:
                            print(f"⚠️ Could not download config after migration: {config_error}")

                    except Exception as migrate_error:
                        print(f"⚠️ Migration failed: {migrate_error}")
            else:
                # Use same directory for non-transformers models
                safetensors_path = model_path.replace(".pt", ".safetensors")
                if not safetensors_path.endswith(".safetensors"):
                    safetensors_path = model_path + ".safetensors"

            # For transformers models in subdirectories, ensure config exists even if model already exists
            if target_subdir and model_key and os.path.exists(safetensors_path):
                try:
                    from engines.rvc.hubert_downloader import _download_transformers_config
                    info = HUBERT_MODELS.get(model_key)
                    if info and info.get('repo_id'):
                        config_dir = os.path.dirname(safetensors_path)
                        config_path = os.path.join(config_dir, 'config.json')
                        if not os.path.exists(config_path):
                            print(f"📥 Safetensors exists but config.json missing, downloading for {model_key}...")
                            _download_transformers_config(info['repo_id'], config_dir)
                except Exception as config_error:
                    print(f"⚠️ Could not download missing config: {config_error}")

            # If safetensors version doesn't exist, create it
            if not os.path.exists(safetensors_path):
                try:
                    import torch
                    from safetensors.torch import save_file

                    print(f"🔧 Converting {model_path} to {safetensors_path}")

                    # Try to extract just the model weights, ignoring fairseq objects
                    try:
                        # First attempt: load with torch pickle_module to handle fairseq objects
                        import pickle
                        import io

                        with open(model_path, 'rb') as f:
                            # Load raw data and try to extract model state_dict only
                            checkpoint = torch.load(f, map_location='cpu', weights_only=False)

                        # Extract state_dict from various possible formats
                        if hasattr(checkpoint, 'state_dict'):
                            state_dict = checkpoint.state_dict()
                        elif isinstance(checkpoint, dict):
                            if 'model' in checkpoint:
                                if hasattr(checkpoint['model'], 'state_dict'):
                                    state_dict = checkpoint['model'].state_dict()
                                else:
                                    state_dict = checkpoint['model']
                            elif 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                # Assume the dict itself is the state_dict
                                state_dict = {k: v for k, v in checkpoint.items()
                                            if isinstance(v, torch.Tensor)}
                        else:
                            raise ValueError("Cannot extract state_dict from checkpoint")

                    except Exception as load_error:
                        print(f"⚠️ Standard loading failed: {load_error}")
                        # Fallback: try to manually extract tensors
                        raise NotImplementedError(f"Cannot convert complex .pt file without fairseq: {model_path}")

                    # Save as safetensors (without config metadata - this is the limitation)
                    save_file(state_dict, safetensors_path)
                    print(f"✅ Converted to safetensors: {safetensors_path}")

                    # For transformers models, download config.json to the subdirectory
                    if target_subdir and model_key:
                        try:
                            from engines.rvc.hubert_downloader import _download_transformers_config
                            info = HUBERT_MODELS.get(model_key)
                            if info and info.get('repo_id'):
                                config_dir = os.path.dirname(safetensors_path)
                                config_path = os.path.join(config_dir, 'config.json')
                                if not os.path.exists(config_path):
                                    print(f"📥 Downloading config.json for {model_key}...")
                                    _download_transformers_config(info['repo_id'], config_dir)
                                else:
                                    print(f"✅ Config already exists: {config_path}")
                        except Exception as config_error:
                            print(f"⚠️ Could not download config: {config_error}")
                    else:
                        print(f"⚠️ Note: Config metadata not preserved - may cause loading issues")

                except Exception as conv_error:
                    print(f"❌ Failed to convert model: {conv_error}")
                    raise NotImplementedError(f"Cannot load .pt file without fairseq: {model_path}")

            # Load the safetensors version
            return HubertModelWithFinalProj.from_safetensors(safetensors_path, device=config.device)
    except Exception as e:
        print(e)
        return None
    
def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    
    # Manual RMS calculation to avoid librosa/numba issues in Python 3.13
    def calculate_rms(y, frame_length, hop_length):
        """Calculate RMS without librosa to avoid numba issues"""
        import numpy as np
        
        # Pad the signal
        y = np.asarray(y)
        n_frames = 1 + (len(y) - frame_length) // hop_length
        
        # Create frames
        frames = np.lib.stride_tricks.sliding_window_view(
            y, window_shape=frame_length
        )[::hop_length]
        
        # Calculate RMS for each frame
        rms = np.sqrt(np.mean(frames**2, axis=1))
        return rms.reshape(1, -1)  # Shape: (1, n_frames)
    
    rms1 = calculate_rms(data1, sr1 // 2 * 2, sr1 // 2)  # 每半秒一个点
    rms2 = calculate_rms(data2, sr2 // 2 * 2, sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2