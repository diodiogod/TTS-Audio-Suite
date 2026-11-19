"""Decode the frozen codec used to tokenize each matched LM's training data.

Decoding goes through a causal multi-resolution iSTFT head declared in the registry entry's ``decoders``
(``codec_registry.json``, istft_head_a_20261002), keyed by decoder id: ``keep2_plain`` (the default,
``decoder_default``: the KEEP-2 pruned head, full-unfreezeall-mr-drop30-4x10) and ``full_mr`` (the full-size head,
full-unfreezeall-mr). The heads share the codec, its frame rate and codebooks and differ only in layer counts. The 600k
conv-tail decoder is never used. The 600k generator.pt only supplies the frozen encoder/quantizer and the initial
decoder stages that the head checkpoint then overrides. ``decoder=None`` is the codec's default decoder everywhere."""

import functools
import json
import os
import time
from pathlib import Path

import torch

from zonos3.utils.hashing import sha256_file
from zonos3.utils.release import hf_join, resolve_file

REGISTRY = Path(__file__).resolve().parents[1] / "registries" / "codec_registry.json"
_CACHE = {}
_INFO = {}
_GRAPHS = {}


@functools.cache
def registry():
    return json.loads(REGISTRY.read_text())


def codec_names():
    return {v["codec"] for v in registry().values()}


def entry(codec):
    values = [v for v in registry().values() if v["codec"] == codec]
    if len(values) != 1:
        raise ValueError(f"Unknown matched codec {codec}")
    return values[0]


def decoder_ids(codec):
    """The ids of the codec's decoders, its default (``decoder_default``) first, then the registry order."""
    decoders = entry(codec).get("decoders") or {}
    default = entry(codec).get("decoder_default")
    if not decoders or default not in decoders:
        raise RuntimeError(
            f"matched codec {codec!r} declares no default iSTFT decoder; the conv-tail decoder is disabled"
        )
    return [default, *(name for name in decoders if name != default)]


def decoder_id(codec, decoder=None):
    """``decoder`` (None = the codec's default) checked against the codec's decoders (ValueError when unknown)."""
    ids = decoder_ids(codec)
    if decoder is None:
        return ids[0]
    if decoder not in ids:
        raise ValueError(f"matched codec {codec!r} has no decoder {decoder!r} (decoders: {', '.join(ids)})")
    return decoder


def served_decoders(codec, wanted=None):
    """The decoder ids a server serves: ``wanted`` (a server flag's ids; None = every decoder of the codec), the
    codec's default first when it is among them, else the first wanted; ValueError for an id the codec lacks."""
    ids = decoder_ids(codec)
    if wanted is None:
        return ids
    wanted = list(dict.fromkeys(str(name).strip() for name in wanted if str(name).strip()))
    if not wanted:
        raise ValueError(f"no decoder given (matched codec {codec!r} has {', '.join(ids)})")
    for name in wanted:
        decoder_id(codec, name)
    return sorted(wanted, key=lambda name: name != ids[0])


def decoder_spec(codec, decoder=None):
    """A registry decoder block; every matched codec decoder must be a causal iSTFT head (no conv-tail fallback)."""
    name = decoder_id(codec, decoder)
    spec = entry(codec)["decoders"][name]
    if spec.get("kind") != "istft_mr":
        raise RuntimeError(
            f"decoder {name!r} of matched codec {codec!r} is no iSTFT head; the conv-tail decoder is disabled"
        )
    return spec


def decoder_headroom_db(codec, decoder=None):
    return float(decoder_spec(codec, decoder).get("headroom_db", 0.0))


def decoder_info(codec, decoder=None):
    name = decoder_id(codec, decoder)
    spec = decoder_spec(codec, name)
    info = {
        "id": name,
        "kind": spec["kind"],
        "label": spec.get("label"),
        "checkpoint": hf_join(spec["campaign"], spec["checkpoint"]),
        "expect_step": spec.get("expect_step"),
        "headroom_db": float(spec.get("headroom_db", 0.0)),
        "causal": True,
    }
    info.update(_INFO.get((codec, name), {}))
    return info


def load_model(codec, device, prune=True, decoder=None):
    """Build the iSTFT-head decoder exactly as the listening viewer's decode.py does: ISTFTCodec(cfg['model'], cfg['head']),
    pretrained generator weights, then the run's trainable state (decoder stages + head) from the pinned checkpoint."""
    e = entry(codec)
    name = decoder_id(codec, decoder)
    spec = decoder_spec(codec, name)
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec

    campaign = spec["campaign"]
    cfg = json.loads(Path(resolve_file(hf_join(campaign, spec["config"]))).read_text())
    assert cfg.get("decoder", {}).get("type") != "original", (
        "the registry points at a conv-tail (original decoder) config"
    )
    generator_path = Path(resolve_file(cfg["init"]["generator_checkpoint"]))
    assert str(generator_path) == resolve_file(e["checkpoint"]), (str(generator_path), e["checkpoint"])
    generator_state = torch.load(generator_path, map_location="cpu", mmap=True, weights_only=True)
    assert generator_state["step"] == e["codec_step"] and generator_state["config"]["model"] == cfg["model"]
    model = ISTFTCodec(cfg["model"], cfg["head"])
    init = model.load_pretrained(generator_state["generator"])
    checkpoint = Path(resolve_file(hf_join(campaign, spec["checkpoint"])))
    # TTS Audio Suite Patch: weights_only=True (was False)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if spec.get("expect_step") is not None:
        assert int(state["step"]) == int(spec["expect_step"]), (
            str(checkpoint),
            int(state["step"]),
            spec["expect_step"],
        )
    if "trainable" in state:
        model.load_trainable_state_dict(state["trainable"])
    else:
        assert not model.trainable_stages
        model.head.load_state_dict(state["head"], strict=True)
    model.eval().requires_grad_(False)
    if prune:
        prune_model(model, e)
    _INFO[codec, name] = {
        "step": int(state["step"]),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(checkpoint.stat().st_mtime)),
        "model": type(model).__name__,
        "init": init if isinstance(init, dict) else None,
    }
    del state, generator_state
    return model.to(device)


def prune_model(model, e):
    # ISTFTCodec never holds the encoder (its __init__ keeps only stages/quantizer/semantic_head/to_external of the base codec);
    # semantic_head is used by features() and must stay.
    for name in ("encoder", "to_internal", "semantic_encoder"):
        if hasattr(model, name):
            try:
                delattr(model, name)
            except Exception:
                pass


def decode_model(model, codes, e):
    """[B, frames, codebooks] int -> float32 [B, samples]; fp32 with TF32 off, like the viewers' decode service."""
    device = next(model.parameters()).device
    packed = codes.permute(0, 2, 1).to(device=device, dtype=torch.int64)
    expected = packed.shape[-1] * e["hop"]
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            audio = model(packed)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    if audio.shape != (packed.shape[0], 1, expected):
        raise RuntimeError(f"Codec output shape {tuple(audio.shape)}; expected {expected} samples")
    if not torch.isfinite(audio).all():
        raise RuntimeError("Nonfinite codec audio")
    return audio.float().squeeze(1).cpu()


def _key(codec, decoder):
    return codec, decoder_id(codec, decoder), os.environ.get("ZONOS3_AUDIO_CODEC_DEVICE", "cuda:0")


def _model(codec, decoder=None):
    """A decoder of the codec on ZONOS3_AUDIO_CODEC_DEVICE (default cuda:0), loaded once after the checkpoint's sha256."""
    e = entry(codec)
    key = _key(codec, decoder)
    if key not in _CACHE:
        actual = sha256_file(resolve_file(e["checkpoint"]))
        assert actual == e["checkpoint_sha256"], ("generator checkpoint sha256", actual, e["checkpoint_sha256"])
        _CACHE[key] = load_model(codec, key[2], decoder=key[1])
    return _CACHE[key]


def decode(codes, codec, decoder=None):
    e = entry(codec)
    if codes.ndim != 3 or codes.shape[-1] != e["codebooks"] or bool(((codes < 0) | (codes >= 4096)).any()):
        raise ValueError("Invalid matched codec tokens")
    return decode_model(_model(codec, decoder), codes, e)


def stream(codec, decoder=None):
    """A fresh stateful decoder (ISTFTStream) on the cached model: decode(codes [n, codebooks]) -> final samples, flush().
    It replays the CUDA graphs captured by ``capture_stream_graphs`` for that decoder and matching chunk shapes."""
    from zonos3.codec.istft_head_a_20261002.stream import ISTFTStream

    return ISTFTStream(_model(codec, decoder), graphs=_GRAPHS.get(_key(codec, decoder)))


def capture_stream_graphs(codec, chunks, decoder=None) -> None:
    """CUDA-graph the chunk shapes of a stream decoding ``chunks`` frames per call from the start (bit-identical to
    the eager decode; a no-op unless the codec runs on CUDA). Each decoder keeps its own graphs."""
    from zonos3.codec.istft_head_a_20261002.stream import ChunkGraphs

    model = _model(codec, decoder)
    if next(model.parameters()).device.type != "cuda":
        return
    _GRAPHS.setdefault(_key(codec, decoder), ChunkGraphs()).capture(model, chunks)
