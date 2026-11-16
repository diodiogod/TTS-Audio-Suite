"""Waveform encoding uses exactly the training SSL feature preprocessor."""

import torch

from zonos3.codec.s1dac.same_data.npy_ogg_dataset import Wav2Vec2BertFeaturePreprocessor


@torch.no_grad()
def extract_ssl(audio, teacher):
    preprocessor = Wav2Vec2BertFeaturePreprocessor(44100)
    items = []
    for example in audio.detach().float().cpu():
        info = {}
        preprocessor(example, info)
        items.append(info)
    features = torch.stack([item["w2v_bert_input_features"] for item in items]).to(audio.device)
    masks = torch.stack([item["w2v_bert_attention_mask"] for item in items]).to(audio.device)
    return teacher(features, masks)


@torch.no_grad()
def encode_audio(model, audio, teacher):
    if model.training:
        raise ValueError("Set the codec to eval mode before token encoding")
    ssl = extract_ssl(audio, teacher)
    return model.quantize_dual(model.encode_latent(audio), ssl)["codes"]
