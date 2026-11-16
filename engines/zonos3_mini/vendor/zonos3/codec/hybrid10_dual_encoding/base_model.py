"""MOSS-v2-style causal transformer codec with the existing semantic-first RVQ.

Architecture reference: OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 (Apache-2.0).
This training implementation retains the repository's attention/FFN dropout.
Boundary sinc conversion has finite FIR lookahead; waveform timestamps are
delay-compensated offline. Patch/attention processing is causal in frame time.
"""

import math

import torch
from dac.nn.layers import Snake1d
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchaudio.transforms import Resample

from zonos3.codec.s1dac.vendor.modded_dac import CausalWNConv1d, DecoderBlock, EncoderBlock
from zonos3.codec.s1dac.vendor.rvq import DownsampleResidualVectorQuantize


class StableSnake1d(Snake1d):
    """Identical S1 Snake math without JIT profiling changes during recompute."""

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


def stable_snakes(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Snake1d):
            replacement = StableSnake1d(child.alpha.shape[1])
            replacement.alpha = child.alpha
            setattr(module, name, replacement)
        else:
            stable_snakes(child)


class CheckpointModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.module, x, use_reentrant=False, preserve_rng_state=True)
        return self.module(x)


class Patch(nn.Module):
    def __init__(self, size, up=False):
        super().__init__()
        self.size, self.up = size, up

    def forward(self, x):
        b, c, t = x.shape
        h = self.size
        if self.up:
            return x.reshape(b, c // h, h, t).permute(0, 1, 3, 2).reshape(b, c // h, t * h)
        return x.reshape(b, c, t // h, h).permute(0, 1, 3, 2).reshape(b, c * h, t // h)


class MossLayer(nn.Module):
    def __init__(self, spec):
        super().__init__()
        d = spec["d_model"]
        self.heads = spec["num_heads"]
        self.head_dim = d // self.heads
        self.context = spec["context_frames"]
        self.attention_dropout = spec["attention_dropout"]
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attention_out = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d, spec["dim_feedforward"], bias=False),
            nn.GELU(),
            nn.Dropout(spec["ffn_dropout"]),
            nn.Linear(spec["dim_feedforward"], d, bias=False),
        )
        self.attention_scale = nn.Parameter(torch.full((d,), spec["layer_scale"]))
        self.ffn_scale = nn.Parameter(torch.full((d,), spec["layer_scale"]))

    def attention(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        # The released MOSS implementation rotates adjacent real/imag pairs
        # in FP32 before returning Q/K to the compute dtype.
        frequency = torch.exp(
            torch.arange(self.head_dim // 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) * 2 / self.head_dim)
        )
        phase = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * frequency[None, :]
        cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]

        def rope(z):
            zf = z.float().reshape(b, t, self.heads, self.head_dim // 2, 2)
            r, i = zf.unbind(-1)
            return torch.stack((r * cos - i * sin, r * sin + i * cos), -1).flatten(-2).to(z.dtype)

        q, k = rope(q), rope(k)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        outputs = []
        dropout = self.attention_dropout if self.training else 0.0
        # Bounded query/key blocks avoid a quadratic 4,800 x 4,800 bias and
        # support the existing dropout on PyTorch's native SDPA backends.
        for start in range(0, t, 512):
            end = min(t, start + 512)
            left = max(0, start - self.context + 1)
            qp = torch.arange(start, end, device=x.device)
            kp = torch.arange(left, end, device=x.device)
            delta = qp[:, None] - kp[None, :]
            valid = (delta >= 0) & (delta < self.context)
            # Padded row stride satisfies efficient-attention bias alignment.
            width = end - left
            bias = x.new_full((end - start, (width + 7) // 8 * 8), float("-inf"))
            bias[:, :width].masked_fill_(valid, 0)
            outputs.append(
                F.scaled_dot_product_attention(
                    q[:, :, start:end],
                    k[:, :, left:end],
                    v[:, :, left:end],
                    attn_mask=bias[:, :width],
                    dropout_p=dropout,
                )
            )
        return self.attention_out(torch.cat(outputs, dim=2).transpose(1, 2).reshape(b, t, d))

    def forward(self, x):
        x = x + self.attention_scale * self.attention(self.norm1(x))
        return x + self.ffn_scale * self.ffn(self.norm2(x))


class MossStage(nn.Module):
    def __init__(self, spec, checkpoint_layers):
        super().__init__()
        self.input_proj = nn.Linear(spec["input_dimension"], spec["d_model"], bias=False)
        self.layers = nn.ModuleList(MossLayer(spec) for _ in range(spec["num_layers"]))
        self.output_proj = nn.Linear(spec["d_model"], spec["output_dimension"], bias=False)
        self.checkpoint_layers = checkpoint_layers

    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2))
        for layer in self.layers:
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)
            else:
                # Avoid new compiled graphs for variable-length evaluation.
                x = layer(x) if self.training else layer.forward(x)
        return self.output_proj(x).transpose(1, 2)


def build_stages(specs, checkpoint_layers):
    modules = []
    for s in specs:
        kind = s["type"]
        if kind == "PatchedPretransform":
            m = Patch(s["patch_size"], s["direction"] == "up")
        elif kind == "MossProjectedTransformer":
            m = MossStage(s, checkpoint_layers)
        elif kind == "S1CausalWNConv1d":
            m = CausalWNConv1d(s["in_channels"], s["out_channels"], s["kernel_size"], stride=s["stride"])
        elif kind == "S1EncoderBlock":
            m = EncoderBlock(s["out_channels"], stride=s["stride"], causal=True, n_t_layer=0)
        elif kind == "S1DecoderBlock":
            m = DecoderBlock(s["in_channels"], s["out_channels"], stride=s["stride"], causal=True, n_t_layer=0)
        elif kind == "S1WaveformHead":
            from dac.nn.layers import Snake1d

            m = nn.Sequential(Snake1d(s["in_channels"]), CausalWNConv1d(s["in_channels"], 1, 7), nn.Tanh())
        else:
            raise ValueError(f"Unknown stage {kind}")
        if kind.startswith("S1"):
            stable_snakes(m)
        if checkpoint_layers and kind in ["S1EncoderBlock", "S1DecoderBlock"]:
            from zonos3.codec.s1dac.vendor.modded_dac import ResidualUnit

            for index, child in enumerate(m.block):
                if isinstance(child, ResidualUnit):
                    m.block[index] = CheckpointModule(child)
            m = CheckpointModule(m)
        modules.append(m)
    return nn.Sequential(*modules)


class MossCodec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = 44100
        self.frame_length = 2940
        self.internal_hop = 2944
        self.encoder = build_stages(cfg["encoder_stages"], cfg["checkpoint_transformers"])
        self.decoder = build_stages(cfg["decoder_stages"], cfg["checkpoint_transformers"])
        self.quantizer = DownsampleResidualVectorQuantize(
            input_dim=1024,
            n_codebooks=cfg["residual_codebooks"],
            codebook_size=cfg["residual_codebook_size"],
            semantic_codebook_size=cfg["semantic_codebook_size"],
            codebook_dim=cfg["codebook_dim"],
            quantizer_dropout=cfg["quantizer_dropout"],
            downsample_factor=(),
        )
        self.semantic_head = nn.Conv1d(1024, 1024, 1)
        self.to_internal = Resample(44100, 44160, lowpass_filter_width=16, rolloff=0.99)
        self.to_external = Resample(44160, 44100, lowpass_filter_width=16, rolloff=0.99)

    def compile(self, **kwargs):
        # Compile reusable transformer blocks, not one enormous 184-layer
        # graph. Module.compile preserves original checkpoint state-dict keys.
        # Six resolutions have BF16 first-layer and FP32 residual inputs.
        # The default eight-specialization cap would leave later stages eager.
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 32)
        for module in self.modules():
            if isinstance(module, MossLayer):
                module.compile(dynamic=False, **kwargs)

    def encode_latent(self, audio):
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("Expected mono audio [batch,1,samples]")
        with torch.autocast(audio.device.type, enabled=False):
            internal = self.to_internal(audio.float())
        internal = F.pad(internal, (0, (-internal.shape[-1]) % self.internal_hop))
        return self.encoder(internal)

    def decode_latent(self, z, length=None):
        internal = self.decoder(z)
        with torch.autocast(z.device.type, enabled=False):
            audio = self.to_external(internal.float())
        return audio if length is None else audio[..., :length]

    def forward(self, audio, **kwargs):
        result = self.quantizer(self.encode_latent(audio))
        return {
            "audio": self.decode_latent(result.z, audio.shape[-1]),
            "codes": result.codes,
            "commitment": result.commitment_loss,
            "codebook": result.codebook_loss,
            "semantic_prediction": self.semantic_head(result.semantic_distill_z),
        }

    def decode_tokens(self, codes, length=None):
        count = 1 + self.cfg["residual_codebooks"]
        if codes.ndim != 3 or not 1 <= codes.shape[1] <= count:
            raise ValueError(f"Expected [batch,1..{count},frames]")
        if bool(((codes < 0) | (codes >= 4096)).any()):
            raise ValueError("Code index outside [0,4096)")
        with torch.autocast(codes.device.type, enabled=False):
            z = self.quantizer.semantic_quantizer.from_codes(codes[:, :1])[0]
            if codes.shape[1] > 1:
                z = z + self.quantizer.quantizer.from_codes(codes[:, 1:])[0]
        return self.decode_latent(z, length)
