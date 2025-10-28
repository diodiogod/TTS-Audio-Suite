import numpy as np, torch, sys, os
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal
import os, traceback, librosa
from scipy import signal
from .lib.model_utils import load_hubert, change_rms

# CRITICAL FIX: Disable CUDNN benchmark to prevent VRAM spikes on Python 3.12+
# Same issue as other TTS engines - CUDNN benchmark causes memory allocation spikes
if torch.cuda.is_available() and sys.version_info >= (3, 12):
    torch.backends.cudnn.benchmark = False

# from tqdm import tqdm

from .pitch_extraction import FeatureExtractor

from .lib.audio import MAX_INT16, load_input_audio, remix_audio
from .config import config
from .lib.utils import gc_collect

# torchcrepe = lazyload("torchcrepe")  # Fork Feature. Crepe algo for training and preprocess
# torch = lazyload("torch")
# rmvpe = lazyload("rmvpe")

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class VC(FeatureExtractor):

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        feats = model.extract_features(version=version,**inputs)

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index is not None and big_npy is not None and index_rate > 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            score, ix = index.search(npy, k=1)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        
        p_len = min(audio0.shape[0] // self.window, feats.shape[1])
        
        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

            if protect < 0.5:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                print("vc",feats.shape,pitch.shape,pitchf.shape)
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
                del pitch, pitchf
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        gc_collect()

        return audio1

    def pipeline(self, model, net_g, sid, audio, times, f0_up_key, f0_method, merge_type,
            file_index, index_rate, if_f0, filter_radius, tgt_sr, resample_sr, rms_mix_rate,
            version, protect, crepe_hop_length, f0_autotune, rmvpe_onnx, f0_file=None, f0_min=50, f0_max=1600):
        
        
        index, big_npy = self.load_index(file_index)

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            
            for t in range(self.t_center, audio.shape[0], self.t_center):
                abs_audio_sum = np.abs(audio_sum[t - self.t_query : t + self.t_query])
                min_abs_audio_sum = abs_audio_sum.min()
                opt_ts.append(t - self.t_query + np.where(abs_audio_sum == min_abs_audio_sum)[0][0])

        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        inp_f0 = None

        if f0_file is not None:
            try:
                with open(f0_file.name, "r") as f:
                    inp_f0 = np.array([list(map(float, line.split(","))) for line in f.read().strip("\n").split("\n")], dtype="float32")
            except:
                traceback.print_exc()

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None

        if if_f0:
            pitch, pitchf = self.get_f0(
                audio_pad, f0_up_key, f0_method, merge_type,
                filter_radius, crepe_hop_length, f0_autotune, rmvpe_onnx, inp_f0, f0_min, f0_max)
            p_len = min(pitch.shape[0], pitchf.shape[0])
            pitch = pitch[:p_len].astype(np.int64 if self.device != 'mps' else np.float32)
            pitchf = pitchf[:p_len].astype(np.float32)
            pitch = torch.from_numpy(pitch).to(self.device).unsqueeze(0)
            pitchf = torch.from_numpy(pitchf).to(self.device).unsqueeze(0)

        t2 = ttime()
        times[1] += t2 - t1

        for i, t in enumerate(opt_ts):
            t = t // self.window * self.window
            start = s
            end = t + self.t_pad2 + self.window
            audio_slice = audio_pad[start:end]
            pitch_slice = pitch[:, start // self.window:end // self.window] if if_f0 else None
            pitchf_slice = pitchf[:, start // self.window:end // self.window] if if_f0 else None
            audio_opt.append(self.vc(model, net_g, sid, audio_slice, pitch_slice, pitchf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
            s = t

        audio_slice = audio_pad[t:]
        pitch_slice = pitch[:, t // self.window:] if if_f0 and t is not None else pitch
        pitchf_slice = pitchf[:, t // self.window:] if if_f0 and t is not None else pitchf
        audio_opt.append(self.vc(model, net_g, sid, audio_slice, pitch_slice, pitchf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate < 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)

        audio_max = np.abs(audio_opt).max() / 0.99
        audio_opt = (audio_opt * MAX_INT16 / audio_max).astype(np.int16)

        gc_collect()

        print("Returning completed audio...")
        print("-------------------")
        
        return audio_opt

def get_vc(model_path,file_index=None,config=config,device=None):
    cpt = torch.load(model_path, map_location="cpu", weights_only=False)
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    if version == "v1":
        if if_f0 == 1:
            from .lib.infer_pack.models import SynthesizerTrnMs256NSFsid
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            from .lib.infer_pack.models import SynthesizerTrnMs256NSFsid_nono
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            from .lib.infer_pack.models import SynthesizerTrnMs768NSFsid
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            from .lib.infer_pack.models import SynthesizerTrnMs768NSFsid_nono
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device if device else config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    model_name = os.path.basename(model_path).split(".")[0]
    
    try: #preload file_index
        if os.path.exists(file_index):
            import faiss
            # Suppress verbose faiss loading messages by redirecting stdout
            import sys
            from io import StringIO

            # Temporarily redirect stdout to suppress faiss loading messages
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                index = faiss.read_index(file_index)
            finally:
                # Always restore stdout
                sys.stdout = old_stdout

            big_npy = index.reconstruct_n(0, index.ntotal)
            file_index = index, big_npy
        else:
            file_index = ""

    except Exception as e:
        print(f"Could not open Faiss index file for reading. {e}")
        file_index = ""

    return {"vc": vc, "cpt": cpt, "net_g": net_g, "model_name": model_name,
            "file_index": file_index, "sr": cpt["config"][-1]}

def vc_single(
    cpt=None,
    net_g=None,
    vc=None,
    hubert_model=None,
    sid=0,
    input_audio=None,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="crepe",
    merge_type="median",
    file_index="",  # .index file
    index_rate=.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=.25,
    protect=0.33,
    crepe_hop_length=160,
    f0_autotune=False,
    is_onnx=False,
    config=config,
    hubert_path=None,
    **kwargs #prevents function from breaking
):
    print(f"vc_single unused args: {kwargs}")
    
    try:
        if hubert_model == None:
            hubert_model = load_hubert(hubert_path,config)

        if not (cpt and net_g and vc and hubert_model):
            print(f"❌ vc_single: Missing required components - cpt: {cpt is not None}, net_g: {net_g is not None}, vc: {vc is not None}, hubert_model: {hubert_model is not None}")
            return None

        tgt_sr = cpt["config"][-1]
        version = cpt.get("version", "v1")

        if input_audio is None and input_audio_path is None:
            print("❌ vc_single: No input audio provided")
            return None
            
        f0_up_key = int(f0_up_key)
        
        print(f"🔧 vc_single: Starting conversion - version: {version}, f0_method: {f0_method}")
        
        audio = input_audio[0] if input_audio is not None else load_input_audio(input_audio_path, 16000)
        audio,_ = remix_audio((audio,input_audio[1] if input_audio is not None else 16000), target_sr=16000)
    
    except Exception as e:
        print(f"❌ vc_single preprocessing error: {e}")
        print(f"🔧 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None
        
    try:

        times = [0, 0, 0]
        if_f0 = cpt.get("f0", 1)
        
        print(f"🔧 vc_single: About to call vc.pipeline - if_f0: {if_f0}")
        
        try:
            """
            model, net_g, sid, audio, times, f0_up_key, f0_method,
                file_index, index_rate, if_f0, filter_radius, tgt_sr, resample_sr, rms_mix_rate,
                version, protect, crepe_hop_length, f0_autotune, rmvpe_onnx
            """
            audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method if len(f0_method)>1 else f0_method[0], # more than 1 f0_method in list means hybrid
            merge_type,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length, f0_autotune, is_onnx,
            f0_file=f0_file,
        )
        
            print(f"✅ vc_single: vc.pipeline completed successfully")
            return (audio_opt, resample_sr if resample_sr >= 16000 and tgt_sr != resample_sr else tgt_sr)
        except Exception as pipeline_error:
            print(f"❌ vc_single: Pipeline error: {pipeline_error}")
            print(f"🔧 Pipeline error type: {type(pipeline_error).__name__}")
            import traceback
            traceback.print_exc()
            return None
    except Exception as error:
        print(error)
        return None