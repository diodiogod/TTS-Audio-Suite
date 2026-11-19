"""DualCodec semantic encoding on the unchanged S1/MOSS waveform codec.

The semantic stream follows DualCodec's ConvNeXt encoder -> Q1 -> decoder.
Baseline cosine supervision, feature pooling and codebook settings are retained.
SSL features are required at encoding time; token decoding is self-contained.
"""

import torch
from dac.nn.layers import WNConv1d
from dac.nn.quantize import ResidualVectorQuantize
from torch import nn
from torch.nn import functional as F

from zonos3.codec.s1dac.losses import average_ssl_frames

from .base_model import MossCodec as BaseCodec


class SemanticConvNeXt(nn.Module):
    """Causal form of DualCodec's depthwise-conv/MLP residual block."""

    def __init__(self, width, hidden):
        super().__init__()
        self.dwconv = nn.Conv1d(width, width, 7, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, hidden)
        self.pwconv2 = nn.Linear(hidden, width)

    def forward(self, x):
        h = self.dwconv(F.pad(x, (6, 0))).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        return x + h.transpose(1, 2)


class MossCodec(BaseCodec):
    def __init__(self, cfg):
        # Preserve construction/initialization of the waveform branches and RVQ-rest.
        super().__init__(cfg)
        branch = cfg["dual_encoding"]
        width, hidden, depth = branch["width"], branch["hidden"], branch["depth"]
        self.semantic_encoder = nn.Sequential(
            WNConv1d(1024, width, 1), *[SemanticConvNeXt(width, hidden) for _ in range(depth)]
        )
        self.semantic_head = nn.Sequential(
            *[SemanticConvNeXt(width, hidden) for _ in range(depth)], WNConv1d(width, 1024, 1)
        )
        self.quantizer.semantic_quantizer = ResidualVectorQuantize(
            input_dim=width,
            n_codebooks=1,
            codebook_size=cfg["semantic_codebook_size"],
            codebook_dim=cfg["codebook_dim"],
            quantizer_dropout=0.0,
        )
        self.quantizer.semantic_quantizer.apply(self.quantizer._init_weights)

    def quantize_dual(self, acoustic, ssl_features, n_quantizers=None):
        if ssl_features.ndim != 3 or ssl_features.shape[0] != acoustic.shape[0] or ssl_features.shape[2] != 1024:
            raise ValueError("Expected SSL features [batch,frames,1024]")
        pooled = average_ssl_frames(ssl_features, acoustic.shape[-1], self.cfg["dual_encoding"]["window_ratio"])
        semantic = self.semantic_encoder(pooled)
        with torch.autocast(acoustic.device.type, enabled=False):
            q1, c1, _, commitment1, codebook1 = self.quantizer.semantic_quantizer(semantic.float())
        semantic = self.semantic_head(q1)
        with torch.autocast(acoustic.device.type, enabled=False):
            residual = acoustic.float() - semantic.float()
            qr, cr, _, commitmentr, codebookr = self.quantizer.quantizer(residual, n_quantizers=n_quantizers)
            z = semantic.float() + qr
        return {
            "z": z,
            "codes": torch.cat([c1, cr], dim=1),
            "semantic_prediction": semantic,
            "commitment": commitment1 + commitmentr,
            "codebook": codebook1 + codebookr,
        }

    def forward(self, audio, ssl_features=None, **kwargs):
        if ssl_features is None:
            raise ValueError(
                "Dual encoding requires frozen SSL features; use inference.encode_audio for waveform input"
            )
        output = self.quantize_dual(self.encode_latent(audio), ssl_features, **kwargs)
        output["audio"] = self.decode_latent(output.pop("z"), audio.shape[-1])
        return output

    def decode_tokens(self, codes, length=None):
        count = 1 + self.cfg["residual_codebooks"]
        if codes.ndim != 3 or not 1 <= codes.shape[1] <= count:
            raise ValueError(f"Expected [batch,1..{count},frames]")
        limits = torch.tensor(
            [self.cfg["semantic_codebook_size"]] + [self.cfg["residual_codebook_size"]] * (codes.shape[1] - 1),
            device=codes.device,
        ).view(1, -1, 1)
        if bool(((codes < 0) | (codes >= limits)).any()):
            raise ValueError("Code index outside its codebook vocabulary")
        with torch.autocast(codes.device.type, enabled=False):
            q1 = self.quantizer.semantic_quantizer.from_codes(codes[:, :1])[0]
        semantic = self.semantic_head(q1)
        with torch.autocast(codes.device.type, enabled=False):
            z = semantic.float()
            if codes.shape[1] > 1:
                z = z + self.quantizer.quantizer.from_codes(codes[:, 1:])[0]
        return self.decode_latent(z, length)
