"""Option A: frozen fixed-600k decoder stages 0-8 (15 Hz -> 240 Hz transformers) + a trainable iSTFT head.

The model consumes cached codes only. There is no encoder, no SSL teacher and no waveform-side conv tail:
decoder.8's 768->5888 output projection, the x23 patch, both S1 conv blocks and the Snake waveform head are
replaced by a small causal ConvNeXt stack, a LayerNorm, a Linear to n_fft+2 channels and an inverse STFT with
hop 184 = 44160/240 and n_fft 736 = 4*hop (periodic Hann, centered frames). Everything except `head.*` is frozen.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from zonos3.codec.hybrid10_dual_encoding.base_model import MossLayer, MossStage, Patch
from zonos3.codec.hybrid10_dual_encoding.model import MossCodec, SemanticConvNeXt

KEPT_STAGES = 9  # decoder.0 .. decoder.8
HEAD_FRAME_RATE = 240


class ISTFTHead(nn.Module):
    def __init__(self, dim, depth, hidden, n_fft, hop):
        super().__init__()
        if n_fft % hop or n_fft // hop < 2:
            raise ValueError("n_fft must be an integer multiple (>= 2) of the hop")
        self.n_fft, self.hop = int(n_fft), int(hop)
        self.blocks = nn.Sequential(*[SemanticConvNeXt(dim, hidden) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.out = nn.Linear(dim, self.n_fft + 2)
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True), persistent=False)
        self.max_log_magnitude = math.log(100.0)  # Vocos clips the magnitude at 1e2

    def spectrum(self, x):
        """x [batch, dim, frames] -> magnitude, phase [batch, n_fft//2+1, frames] (fp32)."""
        h = self.blocks(x)
        y = self.out(self.norm(h.transpose(1, 2)))
        log_magnitude, phase = y.float().chunk(2, dim=-1)
        magnitude = torch.exp(log_magnitude.clamp(max=self.max_log_magnitude))
        return magnitude.transpose(1, 2), phase.transpose(1, 2)

    def forward(self, x):
        magnitude, phase = self.spectrum(x)
        with torch.autocast(x.device.type, enabled=False):
            spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
            audio = torch.istft(
                spec,
                self.n_fft,
                hop_length=self.hop,
                win_length=self.n_fft,
                window=self.window,
                center=True,
                length=magnitude.shape[-1] * self.hop,
            )
        return audio.unsqueeze(1)


class MultiResISTFTHead(nn.Module):
    """Multi-resolution iSTFT head: one causal ConvNeXt trunk, one linear projection per window size, each synthesised
    with its own centered iSTFT at the shared hop, then a linear FIR mixer (fp32, no nonlinearity) combines the branches.

    Compared with ISTFTHead: a deeper trunk, a long window branch (harmonic resolution: 2208 = 12 x 184 gives 20 Hz bins
    at 50 ms) next to the short one (736: 60 Hz bins at 16.7 ms, transients), and a learned linear post-filter. Nothing
    after the synthesis is nonlinear, so the output band above the features' content is a linear function of them.

    Warm start: `blocks.{0..k}` and `norm` match ISTFTHead's names; blocks beyond the warm-started ones are zero-initialised
    at pwconv2 (identity at start); the mixer starts as a delta on branch 0 and zeros on the others, so at step 0 the
    output equals the branch-0 head alone and the extra branches are switched in by gradient (zero-init gating)."""

    def __init__(self, dim, depth, hidden, n_ffts, hop, fir_kernel=127, warm_blocks=None):
        super().__init__()
        self.hop = int(hop)
        self.n_ffts = [int(n) for n in n_ffts]
        for n in self.n_ffts:
            if n % self.hop or n // self.hop < 2:
                raise ValueError("every n_fft must be an integer multiple (>= 2) of the hop")
        self.blocks = nn.Sequential(*[SemanticConvNeXt(dim, hidden) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.outs = nn.ModuleList([nn.Linear(dim, n + 2) for n in self.n_ffts])
        for i, n in enumerate(self.n_ffts):
            self.register_buffer(f"window_{i}", torch.hann_window(n, periodic=True), persistent=False)
        if fir_kernel % 2 == 0:
            raise ValueError("fir_kernel must be odd (zero-phase at init)")
        self.mixer = nn.Conv1d(len(self.n_ffts), 1, fir_kernel, padding=(fir_kernel - 1) // 2, bias=False)
        with torch.no_grad():
            self.mixer.weight.zero_()
            self.mixer.weight[0, 0, (fir_kernel - 1) // 2] = 1.0  # branch 0 passes through, other branches gated off
        if warm_blocks is not None:
            for block in list(self.blocks)[int(warm_blocks) :]:
                nn.init.zeros_(block.pwconv2.weight)
                nn.init.zeros_(block.pwconv2.bias)
        self.max_log_magnitude = math.log(100.0)

    def spectra(self, x):
        """x [batch, dim, frames] -> list of (magnitude, phase) [batch, n_fft//2+1, frames] (fp32)."""
        h = self.norm(self.blocks(x).transpose(1, 2))
        out = []
        for lin in self.outs:
            y = lin(h)
            log_magnitude, phase = y.float().chunk(2, dim=-1)
            magnitude = torch.exp(log_magnitude.clamp(max=self.max_log_magnitude))
            out.append((magnitude.transpose(1, 2), phase.transpose(1, 2)))
        return out

    def forward(self, x):
        frames = x.shape[-1]
        branches = []
        with torch.autocast(x.device.type, enabled=False):
            for i, (magnitude, phase) in enumerate(self.spectra(x)):
                spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
                n = self.n_ffts[i]
                branches.append(
                    torch.istft(
                        spec,
                        n,
                        hop_length=self.hop,
                        win_length=n,
                        window=getattr(self, f"window_{i}"),
                        center=True,
                        length=frames * self.hop,
                    )
                )
            stacked = torch.stack(branches, dim=1)  # [batch, branches, samples]
            return self.mixer(stacked.float())  # [batch, 1, samples], linear, fp32


class MossTailStage(MossStage):
    """MossStage whose output projection (the waveform samples per frame) runs in fp32 outside autocast; everything else
    (input projection, RoPE causal local attention layers, LayerScale, dropout, optional per-layer recompute) is the stock
    MossStage used by decoder.0-8."""

    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2))
        for layer in self.layers:
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = layer(x) if self.training else layer.forward(x)
        with torch.autocast(x.device.type, enabled=False):
            return self.output_proj(x.float()).transpose(1, 2)


class MossTailHead(nn.Module):
    """MOSS-v2 decoder tail in place of the iSTFT head (2026-10-08).

    Mirrors the last two modules of the OpenMOSS MOSS-Audio-Tokenizer-v2 decoder at this codec's rates, as in the
    s1_frontend_moss_v2 proposal (moss_v2_proposals_20260904): decoder.8's output projection (768 -> 768, the stage-2
    `output_proj` the hybrid codec replaced with a 5888-wide conv adapter), PatchedPretransform x2 up (240 Hz, 768 ch ->
    480 Hz, 384 ch), one MossProjectedTransformer stage (d 768, 12 layers, 12 heads, ffn 3072, 1 s causal local context,
    RoPE, LayerScale 0.01, dropout 0.1) projecting to 92 waveform samples per 480 Hz frame, PatchedPretransform x92 up
    -> 44160 Hz waveform. No iSTFT, no Snake, no tanh. All parameters are fresh (MOSS from-scratch init: default Linear
    init, LayerScale 0.01 so the stage starts as a linear map from the 240 Hz features to the waveform)."""

    def __init__(self, dim, hop, spec, patch_in=2):
        super().__init__()
        spec = dict(spec)
        self.patch_in = int(patch_in)
        self.samples_per_frame = int(spec["output_dimension"])
        if spec["input_dimension"] * self.patch_in != dim:
            raise ValueError(f"stage input_dimension x patch_in must equal the 240 Hz width {dim}")
        if self.samples_per_frame * self.patch_in != hop:
            raise ValueError(f"stage output_dimension x patch_in must equal the 240 Hz hop {hop}")
        if spec.get("frame_rate") != HEAD_FRAME_RATE * self.patch_in:
            raise ValueError(f"stage frame_rate must be {HEAD_FRAME_RATE * self.patch_in}")
        self.proj = nn.Linear(dim, dim, bias=False)  # decoder.8.output_proj as in the MOSS decoder (768 -> 768)
        self.up = Patch(self.patch_in, up=True)  # [b, 768, t] -> [b, 384, 2t]
        self.stage = MossTailStage(
            spec, checkpoint_layers=False
        )  # input_proj 384 -> 768, 12 layers, output_proj 768 -> 92
        self.unpatch = Patch(self.samples_per_frame, up=True)  # [b, 92, 2t] -> [b, 1, 184t]

    def forward(self, x):
        """x [batch, 768, frames @ 240 Hz] -> audio [batch, 1, frames * 184] at 44160 Hz (fp32)."""
        h = self.proj(x.transpose(1, 2)).transpose(1, 2)
        h = self.stage(self.up(h))
        return self.unpatch(h)


class ISTFTCodec(nn.Module):
    """Frozen codes -> latent -> decoder.0..8 features at 240 Hz; trainable head -> 44160 Hz -> 44100 Hz."""

    def __init__(self, cfg, head_cfg):
        super().__init__()
        stages = cfg["decoder_stages"]
        assert stages[KEPT_STAGES - 1]["type"] == "MossProjectedTransformer"
        assert stages[KEPT_STAGES - 1]["frame_rate"] == HEAD_FRAME_RATE
        assert stages[KEPT_STAGES]["type"] == "PatchedPretransform" and stages[KEPT_STAGES]["patch_size"] == 23
        hop = cfg["internal_sample_rate"] // HEAD_FRAME_RATE
        assert hop * HEAD_FRAME_RATE == cfg["internal_sample_rate"] and cfg["internal_hop_length"] == hop * (
            HEAD_FRAME_RATE // cfg["frame_rate"]
        )
        base = MossCodec(cfg)
        kept = list(base.decoder)[:KEPT_STAGES]
        kept[-1].output_proj = nn.Identity()  # the 768->5888 projection only fed the removed patch/conv tail
        for stage in kept:
            if isinstance(stage, MossStage):
                stage.checkpoint_layers = False
        # Optional layer pruning of kept transformer stages, e.g. {"0": [0..7, 24..31]} keeps 16 of decoder.0's 32 layers
        # (the middle 16 removed). load_pretrained maps each kept layer's step-600000 weights onto its new index.
        self.stage_layers = {int(k): [int(i) for i in v] for k, v in head_cfg.get("stage_layers", {}).items()}
        for si, keep in self.stage_layers.items():
            stage = kept[si]
            assert isinstance(stage, MossStage), f"stage {si} is not a transformer stage"
            assert keep == sorted(set(keep)) and keep and all(0 <= i < len(stage.layers) for i in keep), (si, keep)
            stage.layers = nn.ModuleList(stage.layers[i] for i in keep)
        self.stages = nn.Sequential(*kept)
        self.quantizer = base.quantizer
        self.semantic_head = base.semantic_head
        self.to_external = base.to_external
        del base
        self.cfg = cfg
        self.hop = hop
        self.upsample = HEAD_FRAME_RATE // cfg["frame_rate"]  # 16 head frames per code frame
        self.external_hop = cfg["external_samples_per_frame"]
        if head_cfg.get("type", "istft") == "moss_tail":
            self.head = MossTailHead(
                dim=stages[KEPT_STAGES - 1]["d_model"],
                hop=hop,
                spec=head_cfg["stage"],
                patch_in=head_cfg.get("patch_in", 2),
            )
        elif head_cfg.get("type", "istft") == "mr_istft":
            self.head = MultiResISTFTHead(
                dim=stages[KEPT_STAGES - 1]["d_model"],
                depth=head_cfg["depth"],
                hidden=head_cfg["hidden"],
                n_ffts=head_cfg["n_ffts"],
                hop=hop,
                fir_kernel=head_cfg.get("fir_kernel", 127),
                warm_blocks=head_cfg.get("warm_blocks"),
            )
        else:
            self.head = ISTFTHead(
                dim=stages[KEPT_STAGES - 1]["d_model"],
                depth=head_cfg["depth"],
                hidden=head_cfg["hidden"],
                n_fft=head_cfg["n_fft"],
                hop=hop,
            )
        # Optionally unfreeze a suffix of the kept transformer stages (e.g. [8] = the 240 Hz stage) with the head.
        self.trainable_stages = sorted({int(i) for i in head_cfg.get("trainable_stages", [])})
        assert self.trainable_stages == list(range(KEPT_STAGES - len(self.trainable_stages), KEPT_STAGES)), (
            "trainable_stages must be a contiguous suffix of decoder.0..8"
        )
        self.trainable_prefixes = ("head.",) + tuple(f"stages.{i}." for i in self.trainable_stages)
        self.requires_grad_(False)
        self.head.requires_grad_(True)
        for i in self.trainable_stages:
            self.stages[i].requires_grad_(True)
        # Optional activation recompute (per-layer checkpointing) inside the trainable transformer stages only.
        self.recompute_trainable_stages = bool(head_cfg.get("recompute_trainable_stages", False))
        if self.recompute_trainable_stages:
            for i in self.trainable_stages:
                if isinstance(self.stages[i], MossStage):
                    self.stages[i].checkpoint_layers = True
            if isinstance(self.head, MossTailHead):
                self.head.stage.checkpoint_layers = True
        self.compile_trainable_stages = bool(head_cfg.get("compile_trainable_stages", True))
        self.train(True)

    def is_trainable_name(self, name):
        return name.startswith(self.trainable_prefixes)

    # frozen modules stay in eval mode (no attention/FFN dropout); only trainable parts follow train()/eval()
    def train(self, mode=True):
        super().train(False)
        self.training = bool(mode)
        self.head.train(mode)
        for i in self.trainable_stages:
            self.stages[i].train(mode)
        return self

    def frozen_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if not self.is_trainable_name(n)]

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if self.is_trainable_name(n)]

    def trainable_named_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if self.is_trainable_name(n)]

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if self.is_trainable_name(k)}

    def load_trainable_state_dict(self, state):
        expected = set(self.trainable_state_dict())
        if set(state) != expected:
            raise RuntimeError(
                f"trainable state keys differ: missing {sorted(expected - set(state))[:5]} unexpected {sorted(set(state) - expected)[:5]}"
            )
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys

    def assert_frozen(self):
        roots = tuple(p.rstrip(".") for p in self.trainable_prefixes)
        for name, p in self.named_parameters():
            assert p.requires_grad == self.is_trainable_name(name), name
            if not self.is_trainable_name(name):
                assert p.grad is None, name
        for name, module in self.named_modules():
            if name and name not in roots and not self.is_trainable_name(name):
                assert not module.training, name

    def load_pretrained(self, generator_state):
        """Load quantizer, semantic head, resampler and decoder.0..8 (minus decoder.8.output_proj) from step 600000."""
        own = self.state_dict()
        loaded = {}
        pruned = 0
        for key, value in generator_state.items():
            if key.startswith("decoder."):
                index = int(key.split(".")[1])
                if index >= KEPT_STAGES or key.startswith(f"decoder.{KEPT_STAGES - 1}.output_proj."):
                    continue
                rest = key[len("decoder.") :]
                if index in self.stage_layers:
                    parts = rest.split(".")
                    if len(parts) > 2 and parts[1] == "layers":
                        old = int(parts[2])
                        if old not in self.stage_layers[index]:
                            pruned += 1
                            continue  # a removed layer: its weights are dropped on purpose
                        parts[2] = str(self.stage_layers[index].index(old))
                        rest = ".".join(parts)
                loaded["stages." + rest] = value
            elif key.startswith(("quantizer.", "semantic_head.", "to_external.")):
                loaded[key] = value
        missing = [k for k in own if k not in loaded and not k.startswith("head.")]
        unexpected = [k for k in loaded if k not in own]
        if missing or unexpected:
            raise RuntimeError(f"pretrained mapping mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        result = self.load_state_dict(loaded, strict=False)
        assert not result.unexpected_keys and all(k.startswith("head.") for k in result.missing_keys)
        out = {
            "loaded_tensors": len(loaded),
            "frozen_parameters": sum(p.numel() for _, p in self.frozen_parameters()),
            "head_parameters": sum(p.numel() for p in self.head.parameters()),
        }
        if self.stage_layers:
            out.update(pruned_tensors=pruned, stage_layers={str(k): v for k, v in self.stage_layers.items()})
        return out

    def compile_stages(self):
        """Compile each frozen transformer layer's forward (MossStage calls layer.forward in eval mode)."""
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 32)
        count = 0
        targets = list(self.stages)
        if isinstance(self.head, MossTailHead) and self.compile_trainable_stages:
            targets.append(self.head.stage)  # the trainable MOSS tail layers compile like the unfrozen stages
        for i, stage in enumerate(targets):
            if i in self.trainable_stages and not self.compile_trainable_stages:
                continue
            for module in stage.modules():
                if isinstance(module, MossLayer):
                    module.forward = torch.compile(module.forward, dynamic=False)
                    count += 1
        return count

    @torch.no_grad()
    def latent(self, codes, counts=None):
        """codes [batch, 10, frames] (long); counts [batch] = residual codebooks kept (1..9), None = all.

        Mirrors MossCodec.decode_tokens / the viewer's decode path: fp32 RVQ lookups, fp32 semantic head.
        Residual codebooks beyond `counts` are dropped per sample, reproducing the training-time codebook
        dropout augmentation on cached codes."""
        if codes.ndim != 3 or codes.shape[1] != 1 + self.cfg["residual_codebooks"]:
            raise ValueError("Expected codes [batch, 10, frames]")
        q = self.quantizer
        with torch.autocast(codes.device.type, enabled=False):
            q1 = q.semantic_quantizer.from_codes(codes[:, :1])[0]
            z = self.semantic_head(q1).float()
            for i, vq in enumerate(q.quantizer.quantizers):
                z_i = vq.out_proj(vq.decode_code(codes[:, 1 + i]))
                if counts is not None:
                    z_i = z_i * (counts > i).to(z_i.dtype)[:, None, None]
                z = z + z_i
        return z

    def features(self, codes, counts=None):
        """240 Hz features [batch, 768, 16*frames] in bf16 autocast; frozen stages run without a graph."""
        x = self.latent(codes, counts)
        with torch.autocast(codes.device.type, dtype=torch.bfloat16, enabled=codes.device.type == "cuda"):
            for i, stage in enumerate(self.stages):
                if i in self.trainable_stages:
                    x = stage(x)
                else:
                    with torch.no_grad():
                        x = stage(x)
        return x

    def forward(self, codes, counts=None, length=None):
        feats = self.features(codes, counts)
        with torch.autocast(codes.device.type, dtype=torch.bfloat16, enabled=codes.device.type == "cuda"):
            internal = self.head(feats)
        with torch.autocast(codes.device.type, enabled=False):
            audio = self.to_external(internal.float())
        expected = codes.shape[-1] * self.external_hop
        audio = audio[..., :expected]
        return audio if length is None else audio[..., :length]
