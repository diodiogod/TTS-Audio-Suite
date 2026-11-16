"""W2V-BERT feature preprocessing of the dual codec's SSL teacher (excerpt of s1dac/same_data/npy_ogg_dataset.py)."""

from functools import lru_cache

import torch
from torchaudio import transforms as T


@lru_cache(maxsize=32)
def _cached_resampler(input_rate: int, output_rate: int) -> T.Resample:
    """Cache polyphase kernels per worker instead of rebuilding every sample."""
    return T.Resample(int(input_rate), int(output_rate))


class Wav2Vec2BertFeaturePreprocessor:
    """Exact public W2V-BERT 2.0 fbank/stride-stack preprocessing on CPU."""

    def __init__(self, source_sample_rate: int):
        from transformers import SeamlessM4TFeatureExtractor

        self.source_sample_rate = int(source_sample_rate)
        self.target_sample_rate = 16000
        self.extractor = SeamlessM4TFeatureExtractor(
            feature_size=80,
            sampling_rate=self.target_sample_rate,
            num_mel_bins=80,
            padding_value=1,
            stride=2,
        )
        self.resampler = (
            _cached_resampler(self.source_sample_rate, self.target_sample_rate)
            if self.source_sample_rate != self.target_sample_rate
            else None
        )

    def __call__(self, audio: torch.Tensor, info: dict) -> None:
        # W2V-BERT is mono.  The production mono recipe already requests an
        # FFmpeg downmix; mean remains a harmless identity for that one channel
        # and keeps this preprocessor valid for other callers.
        mono = audio.float().mean(dim=0)
        if self.resampler is not None:
            mono = self.resampler(mono)
        encoded = self.extractor(
            mono.numpy(),
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )
        info["w2v_bert_input_features"] = encoded.input_features[0]
        info["w2v_bert_attention_mask"] = encoded.attention_mask[0].to(torch.bool)
