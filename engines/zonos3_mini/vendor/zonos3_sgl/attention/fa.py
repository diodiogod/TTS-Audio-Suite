from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from zonos3.models.config import ModelConfig

    from zonos3_sgl.core import TTSBatch
    from zonos3_sgl.kvcache import BaseKVCache


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


def fa_num_splits(config: ModelConfig) -> int:
    """KV splits for flash_attn_with_kvcache. Training runs flash_attn_varlen_func unsplit (num_splits=1); FA3's
    auto heuristic (0) splits the KV reduction of small grids over long caches (> 512 rows) and combines partial
    softmaxes, which rounds differently. GDN-family checkpoints therefore run unsplit so decode and lone prefills
    stay bit for bit on the training arithmetic (receipt: sgl_frozen_arch_20261012/checks/fa3-vs-training-probe.json);
    legacy global-attention models keep the heuristic. ZONOS3_FA_NUM_SPLITS=<n> forces a value for any model."""
    forced = os.environ.get("ZONOS3_FA_NUM_SPLITS")
    if forced is not None and forced.strip():
        return max(0, int(forced))
    return 1 if str(getattr(config, "attention_pattern", "global")) != "global" else 0


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
        self.config = config
        self.kvcache = kvcache
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.page_table = page_table
        self.num_splits = fa_num_splits(config)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: TTSBatch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            num_splits=self.num_splits,
        )

    def prepare_metadata(self, batch: TTSBatch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **cpu_kwargs)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        positions = make_positions(device, reqs)
        page_table = self.page_table
        new_page_table = torch.stack([page_table[req.table_idx, :max_seqlen_k] for req in reqs])

        # copy from CPU to GPU
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            positions=positions,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int], frame_width: int = 1) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len, self.kvcache.device, frame_width)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: TTSBatch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            positions=capture.positions[:bs],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1),
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]

    def prepare_for_replay(self, batch: TTSBatch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        capture = self.capture

        # Copy all dynamic tensors to capture buffers (fixed memory addresses for CUDA graph)
        capture.input_ids[:bs].copy_(batch.input_ids)
        capture.out_loc[:bs].copy_(batch.out_loc)
        capture.positions[:bs].copy_(metadata.positions)
        capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        capture.page_table[:bs, : metadata.max_seqlen_k].copy_(metadata.page_table)

        # Update metadata and batch to point to capture buffers
        # This ensures CUDA graph uses fixed memory addresses for all inputs
        metadata.positions = capture.positions[:bs]
        metadata.cu_seqlens_k = capture.cu_seqlens_k[: bs + 1]
        metadata.cache_seqlens = capture.seq_lens[:bs]
        metadata.page_table = capture.page_table[:bs, :]
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    """# TTS Audio Suite Patch: sgl_kernel.flash_attn.flash_attn_with_kvcache (FlashAttention-3, a compiled CUDA
    extension) is replaced by a Triton kernel (kernel/triton/paged_attention.py): causal attention of the packed query
    rows over each request's paged KV.

    ``q`` [rows, heads, dim]; ``k_cache`` / ``v_cache`` [pages, 1, kv heads, dim] (page size 1); ``page_table``
    [requests, columns] maps a request's positions to pages; ``cache_seqlens`` [requests] is each request's length with
    its new rows, ``cu_seqlens_q`` [requests + 1] the packed query offsets. Query row r of a request (its last query
    row sits at position length - 1) sees keys at positions <= its own. Nothing is read on the host, so the call is
    CUDA-graph safe. Scores are accumulated in float32."""
    if window_size != (-1, -1) or softcap != 0.0 or not causal:
        raise NotImplementedError("only full causal attention is supported in the TTS Audio Suite copy")
    from zonos3_sgl.kernel.triton.paged_attention import paged_attention

    rows = q.shape[0]
    lengths = cache_seqlens.long()
    offsets = cu_seqlens_q.long()
    row = torch.arange(rows, device=q.device)
    owner = torch.searchsorted(offsets[1:], row, right=True)  # request of every query row
    last = lengths[owner] - (offsets[1:][owner] - row)  # its absolute position in that request
    return paged_attention(
        q.contiguous(),
        k_cache,
        v_cache,
        page_table.contiguous(),
        owner.to(torch.int32),
        last.to(torch.int32),
        softmax_scale,
    )
