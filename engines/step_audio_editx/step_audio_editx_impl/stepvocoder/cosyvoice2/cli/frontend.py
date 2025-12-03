import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable, Dict, Literal
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from stepvocoder.cosyvoice2.matcha.audio import mel_spectrogram


class CosyVoiceFrontEnd(object):
    def __init__(self, 
                 mel_conf:Dict,
                 campplus_model:str,
                 speech_tokenizer_model:str,
                 onnx_provider:str='CUDAExecutionProvider',
                 ):
        super().__init__()
        assert onnx_provider in ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'invalid onnx provider'
        self.mel_conf = mel_conf
        self.sample_rate = mel_conf['sampling_rate']
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option,
            providers=["CPUExecutionProvider"]
        )
    
    def extract_speech_feat(self, audio:torch.Tensor, audio_sr:int):
        if audio_sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sr, new_freq=self.sample_rate)
            audio_sr = self.sample_rate
        speech_feat = mel_spectrogram(y=audio, **self.mel_conf).transpose(1, 2) # (b=1, t, num_mels)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.long)
        return speech_feat, speech_feat_len

    def extract_spk_embedding(self, audio:torch.Tensor, audio_sr:int):
        if audio_sr != 16000:
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sr, new_freq=16000)
            audio_sr = 16000
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        onnx_in = {
            self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
        }
        embedding = self.campplus_session.run(None, onnx_in)[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding