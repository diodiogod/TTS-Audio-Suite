
import hashlib
import torch.nn.functional as F
import librosa
import torch

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
            raise NotImplementedError("Please use content-vec-best.safetensors!")
            # from fairseq import checkpoint_utils
            # models, _, _ = checkpoint_utils.load_model_ensemble_and_task([model_path],suffix="",)
            # hubert_model = models[0]
            # hubert_model = hubert_model.to(config.device)
            # if config.is_half:
            #     hubert_model = hubert_model.half()
            # else:
            #     hubert_model = hubert_model.float()
            # hubert_model.eval()
            # return hubert_model
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