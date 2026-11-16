"""Own-weights MECT-B2 speaker encoder for the `ownmect_s2`-style speaker-embedding codecs (2026-10-16).

Mirrors George's training-side extractor (w2v-bert-sv-mect `local.mect_sv.MECTFbank`, `embed_prompt_scp.py`, infer config
`ownmect_b2_infer.yaml`, `use_amp: false`): waveform -> mono float32 at 16 kHz -> at most 60 s -> 80-d Kaldi fbank
(25 ms / 10 ms, hamming, dither 0, energy floor 0, waveform scaled by 2^15) -> AntSpeaker VPR (per-utterance CMN,
MECT_B2 backbone, ASTP pooling, BatchNorm, linear 192, fp32) -> (1, 192). The vectors equal the `ownmect_s2` column rows /
ZTTS2 prompt vectors the finetunes were trained and evaluated with (receipt: frozen_arch checks/mect-embedder-20261016.json).

Which encoder serves which training codec comes from the speaker encoder registry (JSON, env
ZONOS3_SPEAKER_ENCODER_REGISTRY, default `zonos3/registries/speaker_encoder_registry.json`): `{codec: {type: "mect_fbank",
checkpoint, sha256, backbone_type, backbone_config, pooling, embed_dim, sample_rate, max_seconds, fbank}}`.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from zonos3.utils.hashing import sha256_file

REGISTRY_ENV = "ZONOS3_SPEAKER_ENCODER_REGISTRY"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "registries" / "speaker_encoder_registry.json"
MECT_FBANK = "mect_fbank"


def registry_path() -> Path:
    from zonos3.utils.release import resolve_file  # the registry itself may live on hf://

    return Path(resolve_file(os.environ.get(REGISTRY_ENV) or str(DEFAULT_REGISTRY)))


def load_registry() -> dict:
    path = registry_path()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")}


def registry_entry(codec: str | None) -> dict | None:
    """The registry entry for a training codec name (exact string from the checkpoint's params.json), or None."""
    if not codec:
        return None
    entry = load_registry().get(str(codec))
    if entry is None:
        return None
    return dict(entry, codec=str(codec))


class MECTFbankSpeakerEmbedding(nn.Module):
    """George's MECTFbank inference path: Kaldi fbank front-end + AntSpeaker VPR (MECT_B2 / ASTP), fp32."""

    TYPE = MECT_FBANK

    def __init__(self, entry: dict, device: str | torch.device = "cuda"):
        super().__init__()
        if entry.get("type") != self.TYPE:
            raise ValueError(
                f"speaker encoder entry {entry.get('codec')!r} is of type {entry.get('type')!r}, not {self.TYPE!r}"
            )
        self.codec = str(entry.get("codec", ""))
        self.device = torch.device(device)
        self.sample_rate = int(entry.get("sample_rate", 16_000))
        self.max_seconds = float(entry.get("max_seconds", 60.0))
        self.embedding_dim = int(entry.get("embed_dim", 192))
        fb = dict(entry.get("fbank") or {})
        self.waveform_scale = float(fb.pop("scale", 1 << 15))
        self.fbank_kwargs = {
            "num_mel_bins": int(fb.get("num_mel_bins", 80)),
            "frame_length": float(fb.get("frame_length", 25.0)),
            "frame_shift": float(fb.get("frame_shift", 10.0)),
            "dither": float(fb.get("dither", 0.0)),
            "energy_floor": float(fb.get("energy_floor", 0.0)),
            "window_type": str(fb.get("window_type", "hamming")),
            "sample_frequency": float(self.sample_rate),
        }
        # the vendored AntSpeaker registry: importing .models registers VPR, the MECT_* backbones and the poolings
        import zonos3.models.third_party.antspeaker.models  # noqa: F401
        from zonos3.models.third_party.antspeaker.utils.registry import create_model

        cfg = {
            "type": "VPR",
            "config": {
                "backbone": {
                    "type": str(entry.get("backbone_type", "MECT_B2")),
                    "config": dict(entry.get("backbone_config") or {}),
                },
                "pooling": {"type": str(entry.get("pooling", "ASTP"))},
                "embed_dim": self.embedding_dim,
            },
        }
        self.model = create_model(cfg)  # state-dict keys model.* as in George's MECTFbank wrapper

        checkpoint = entry.get("checkpoint")
        if checkpoint:
            from zonos3.utils.release import resolve_file as _resolve_release_file

            path = Path(_resolve_release_file(str(checkpoint)))  # hf:// -> local cache
            if not path.is_file():
                raise FileNotFoundError(f"speaker encoder checkpoint for {self.codec!r} not found: {path}")
            expected = str(entry.get("sha256") or "").strip().lower()
            if expected:
                actual = sha256_file(path)
                if actual != expected:
                    raise ValueError(f"speaker encoder checkpoint {path} sha256 {actual} != registry {expected}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "modules" in state:
                state = state["modules"]["spk_model"]
            state = {
                k: v for k, v in state.items() if not k.startswith("bfbank.")
            }  # training-only batched fbank buffers
            self.load_state_dict(state, strict=True)
            del state
        self.checkpoint = str(checkpoint or "")

        self.to(self.device)
        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, self.sample_rate).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Expected mono/stereo waveform, got shape {tuple(wav.shape)}.")
        wav = wav.to(self.device, torch.float32)
        if int(sample_rate) != self.sample_rate:
            wav = self._get_resampler(int(sample_rate))(wav)
        return wav[..., : int(self.max_seconds * self.sample_rate)]

    def fbank(self, wav: torch.Tensor) -> torch.Tensor:
        """(T,) float waveform in [-1, 1] -> (T', num_mel_bins) Kaldi fbank, as MECTFbank.fbank."""
        return kaldi.fbank(wav[None] * self.waveform_scale, **self.fbank_kwargs)

    def forward(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = self.prepare_input(wav, int(sample_rate))
        with torch.autocast(device_type="cuda", enabled=False):
            feats = self.fbank(wav[0].float())[None].transpose(1, 2)  # (1, mel, T')
            return self.model(feats).to(torch.float32)  # (1, embed_dim)


def create_registry_speaker_embedder(entry: dict, device: str | torch.device = "cuda") -> nn.Module:
    kind = entry.get("type")
    if kind == MECT_FBANK:
        return MECTFbankSpeakerEmbedding(entry, device=device)
    raise ValueError(f"Unsupported speaker encoder type {kind!r} for codec {entry.get('codec')!r} in {registry_path()}")
