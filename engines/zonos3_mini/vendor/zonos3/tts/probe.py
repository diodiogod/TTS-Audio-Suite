"""Alignment-probe math shared by the servers and the reference: head weights, text windows and the readout.

Positions are indices into the text span: 0 is the BOS row, ``text_len - 1`` the EOS row.
"""

from __future__ import annotations

import numpy as np
import torch

Head = tuple[int, int]  # (layer_id, q_head)

_ATTENTION_LAYERS: set[int] = set()


def register_attention_layer(layer_id: int) -> None:
    _ATTENTION_LAYERS.add(int(layer_id))


def attention_layers() -> tuple[int, ...]:
    return tuple(sorted(_ATTENTION_LAYERS))


def clip_window(window: tuple[int, int], text_len: int) -> tuple[int, int]:
    lo = max(0, min(int(window[0]), text_len - 1))
    return lo, min(text_len - 1, max(int(window[1]), lo))


def locate_text_span(input_ids: torch.Tensor, n_codebooks: int, bos: int = 2, eos: int = 3) -> tuple[int, int]:
    """(start, length) of the BOS..EOS text rows of a prompt, read from the text column (index ``n_codebooks``)."""
    text = input_ids[:, n_codebooks]
    starts = (text == bos).nonzero()
    if starts.numel() == 0:
        raise ValueError(f"no text BOS row (id {bos}) in the prompt")
    start = int(starts[0])
    ends = (text[start + 1 :] == eos).nonzero()
    if ends.numel() == 0:
        raise ValueError(f"no text EOS row (id {eos}) after BOS in the prompt")
    end = start + 1 + int(ends[0])
    if bool((text[start + 1 : end] < 4).any()):
        raise ValueError("a row strictly inside the text span has a special text id (< 4)")
    return start, end - start + 1


def probe_weight(heads: tuple[Head, ...], layers: tuple[int, ...], n_heads: int) -> np.ndarray:
    """[len(layers), n_heads] float32 head weights: 1 on every probed head."""
    w = np.zeros((len(layers), n_heads), dtype=np.float32)
    for layer, head in heads:
        if layer not in layers or not 0 <= head < n_heads:
            raise ValueError(f"align probe head {(layer, head)} is not an attention head of this model")
        w[layers.index(layer), head] = 1.0
    return w


def readout(
    q: torch.Tensor,
    text_keys: torch.Tensor,
    heads: tuple[int, ...],
    weights: np.ndarray | torch.Tensor,
    windows: dict[Head, tuple[int, int]] | None,
    scale: float,
) -> tuple[int, float]:
    """Text-attention readout of one decode row (the single-request form of the batched server readout).

    q [L, H, D]: the row's queries at the attention layers ``heads`` (layer ids, in order); text_keys [L, T, KV, D]:
    the keys of the BOS..EOS text rows at those layers; weights [L, H]: head weights (``probe_weight``); windows:
    per-head inclusive text windows (masked heads read a window-renormalized distribution), or None.
    Returns (argmax, centroid) of the weighted sum of the per-head text distributions."""
    n_layers, n_heads, head_dim = q.shape
    text_len, kv = text_keys.shape[1], text_keys.shape[2]
    w = torch.as_tensor(np.asarray(weights, dtype=np.float32)).to(q.device)
    ar = torch.arange(text_len, device=q.device)
    lo = torch.zeros((n_layers, n_heads), dtype=torch.long, device=q.device)
    hi = torch.full((n_layers, n_heads), text_len - 1, dtype=torch.long, device=q.device)
    for (layer, head), window in (windows or {}).items():
        lo[heads.index(layer), head], hi[heads.index(layer), head] = clip_window(window, text_len)
    acc = torch.zeros(text_len, dtype=torch.float32, device=q.device)
    for s in range(n_layers):
        logits = torch.einsum(
            "kgd,lkd->kgl", q[s].float().view(kv, n_heads // kv, head_dim), text_keys[s].float()
        ).reshape(n_heads, text_len) * float(scale)
        allowed = (ar[None, :] >= lo[s, :, None]) & (ar[None, :] <= hi[s, :, None])
        p = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        acc = acc + (p * w[s, :, None]).sum(0)
    m = (acc * ar).sum() / acc.sum().clamp_min(1e-30)
    return int(acc.argmax()), float(m)
