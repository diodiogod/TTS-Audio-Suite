
import hashlib
import torch.nn.functional as F
import librosa
import torch
import os

from infer_pack.loaders import HubertModelWithFinalProj

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
        if model_path.endswith(".safetensors"):
            return HubertModelWithFinalProj.from_safetensors(model_path, device=config.device)
        else:
            # Convert .pt file to .safetensors format for compatibility
            print(f"🔄 Converting .pt Hubert model to safetensors format: {model_path}")
            
            # Generate safetensors path
            safetensors_path = model_path.replace(".pt", ".safetensors")
            if not safetensors_path.endswith(".safetensors"):
                safetensors_path = model_path + ".safetensors"
            
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
                    
                    # Save as safetensors
                    save_file(state_dict, safetensors_path)
                    print(f"✅ Converted to safetensors: {safetensors_path}")
                    
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
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
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