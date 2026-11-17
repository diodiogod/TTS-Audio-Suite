"""CUDA graphs of prefills on GDN checkpoints.

An eager prefill is launch-bound: ~900 kernels (~1,400 with a reference prefix) for 3-6 ms of GPU work in 40-60 ms.
Here each fresh request is padded to its complete 64-row chunks plus one 64-row window (models.gdn.PrefillLayout), so
the layer stack's shapes depend only on the requests' complete-chunk counts: one graph per tuple of counts, captured on
first use (prewarmed for single requests) and kept in a bounded LRU.

The padded rows follow each request's real rows, and no real row reads them: attention is causal (each request is its
own sequence over its own page-table row; the padded positions' KV goes to the dummy page), GDN zeroes them (the
decode window's layout), and every other kernel is per row and independent of the batch's row count (the exact-mode
contract; a request's prefill logits are bit-identical with and without other rows in its batch). The embedding
prologue (speaker and condition rows) runs eagerly on the real rows, which are then scattered into the padded input.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from zonos3_sgl.attention.fa import FAMetadata
from zonos3_sgl.core import Context, TTSBatch
from zonos3_sgl.models.gdn import CHUNK, OriginalGDN, PrefillInputs, PrefillLayout
from zonos3_sgl.utils.logger import init_logger
from zonos3_sgl.utils.torch_utils import make_device_tensor

if TYPE_CHECKING:
    from zonos3_sgl.models import Zonos3ForCausalLM

logger = init_logger(__name__)


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    # every tensor the captured kernels read besides the runner's input buffers: the graph holds raw pointers to them
    layout: PrefillLayout
    tensors: tuple[torch.Tensor, ...]


class PrefillGraphs:
    def __init__(
        self,
        *,
        model: Zonos3ForCausalLM,
        ctx: Context,
        page_table: torch.Tensor,
        dummy_page: int,
        gdn_layers: list[OriginalGDN],
        stream: torch.cuda.Stream,
        max_requests: int,
        max_rows: int,
        max_graphs: int,
    ) -> None:
        self.model = model
        self.ctx = ctx
        self.page_table = page_table
        self.dummy_page = dummy_page
        self.gdn_layers = gdn_layers
        self.stream = stream
        self.max_requests = max_requests
        self.max_rows = min(max_rows - max_rows % CHUNK, page_table.size(1) - page_table.size(1) % CHUNK)
        self.max_graphs = max(1, max_graphs)
        device = page_table.device
        weight = model.multi_output.weight
        # graph inputs: the embedded rows in padded layout, the requests' slots and real row counts
        self.x = torch.zeros((self.max_rows, weight.shape[1]), dtype=weight.dtype, device=device)
        self.slots = torch.zeros((max_requests,), dtype=torch.long, device=device)
        self.lengths = torch.ones((max_requests,), dtype=torch.long, device=device)
        self.logits: torch.Tensor | None = None  # graph output: [max_requests, n_codebooks, vocab]
        self.graphs: OrderedDict[tuple[int, ...], _Graph] = OrderedDict()
        self.pool = torch.cuda.graph_pool_handle()

    def key(self, batch: TTSBatch) -> tuple[int, ...] | None:
        """Complete 64-row chunks per request, or None when the batch stays eager: continuations (a cached prefix:
        LACI rollbacks), accent adapters (per-row adapter inputs), more rows than the graphs cover."""
        reqs = batch.reqs
        if not reqs or len(reqs) > self.max_requests or batch.accent_slots is not None:
            return None
        if any(r.cached_len for r in reqs):
            return None
        whole = tuple(r.extend_len // CHUNK for r in reqs)
        return whole if CHUNK * sum(w + 1 for w in whole) <= self.max_rows else None

    def prewarm(self, keys: list[tuple[int, ...]], dummy_slot: int) -> None:
        """Capture ``keys`` up front on the dummy request's slot (its page-table row is the dummy page)."""
        for key in keys:
            if key in self.graphs or CHUNK * sum(w + 1 for w in key) > self.max_rows:
                continue
            self.slots[: len(key)].fill_(dummy_slot)
            self.lengths[: len(key)].copy_(torch.tensor([CHUNK * w + 1 for w in key]))
            self._capture(key)
        logger.info_rank0(
            "Prefill CUDA graphs: up to %d rows, %d graphs; prewarmed %s",
            self.max_rows,
            self.max_graphs,
            ", ".join(str(k) for k in self.graphs) or "none",
        )

    def run(self, batch: TTSBatch, key: tuple[int, ...]) -> torch.Tensor:
        """The batch's last-row logits [requests, n_codebooks, vocab] (the eager prefill's ``logits[last rows]``)."""
        reqs = batch.reqs
        lengths = [r.extend_len for r in reqs]
        with self.ctx.forward_batch(batch):
            x = self.model.embed(batch.input_ids)
        destination, start = [], 0
        for w, n in zip(key, lengths):
            destination += range(start, start + n)
            start += CHUNK * (w + 1)
        self.x[:start].zero_()
        self.x.index_copy_(0, make_device_tensor(destination, torch.long, x.device), x)
        inputs = make_device_tensor([r.table_idx for r in reqs] + lengths, torch.long, x.device)
        self.slots[: len(reqs)].copy_(inputs[: len(reqs)])
        self.lengths[: len(reqs)].copy_(inputs[len(reqs) :])
        graph = self.graphs.get(key)
        if graph is None:
            graph = self._capture(key)
        self.graphs.move_to_end(key)
        graph.graph.replay()
        for gdn in self.gdn_layers:
            gdn.note_graph_forward(reqs)
        return self.logits[: len(reqs)]

    def _capture(self, key: tuple[int, ...]) -> _Graph:
        """Warm up (Triton compilation, the AttnRes bank of this row count) on the current inputs,
        then capture the same forward; both write identical states for the requests in the slots."""
        device = self.x.device
        lengths = [CHUNK * (w + 1) for w in key]
        rows, longest = sum(lengths), max(lengths)
        layout = PrefillLayout.build(list(key), device)
        cu_seqlens = make_device_tensor([sum(lengths[:i]) for i in range(len(lengths) + 1)], torch.int32, device)
        seq_lens = make_device_tensor(lengths, torch.int32, device)
        positions = make_device_tensor([p for n in lengths for p in range(n)], torch.int32, device)
        column = positions.long()
        owner = make_device_tensor([i for i, n in enumerate(lengths) for _ in range(n)], torch.long, device)
        starts = make_device_tensor([sum(lengths[:i]) for i in range(len(lengths))], torch.long, device)
        columns = torch.arange(longest, device=device)
        b = len(key)

        def forward() -> torch.Tensor:
            slots, real = self.slots[:b], self.lengths[:b]
            # each request's page-table row up to its real rows, the dummy page after (the padded rows' KV)
            page_table = self.page_table.index_select(0, slots)[:, :longest]
            page_table = torch.where(columns.view(1, -1) < real.view(-1, 1), page_table, self.dummy_page)
            batch = TTSBatch(reqs=[], phase="prefill")
            batch.out_loc = page_table[owner, column]
            batch.attn_metadata = FAMetadata(
                cu_seqlens_k=cu_seqlens,
                cu_seqlens_q=cu_seqlens,
                positions=positions,
                cache_seqlens=seq_lens,
                max_seqlen_k=longest,
                max_seqlen_q=longest,
                page_table=page_table,
            )
            batch.gdn_prefill = PrefillInputs(layout=layout, slots=slots, lengths=real)
            with self.ctx.forward_batch(batch):
                logits = self.model.compute_logits(self.model.forward_layers(self.x[:rows]))
            return logits.index_select(0, starts + real - 1)

        warm = forward()
        if self.logits is None:
            self.logits = torch.empty((self.max_requests, *warm.shape[1:]), dtype=warm.dtype, device=device)
        self.logits[:b].copy_(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
            self.logits[:b].copy_(forward())
        static = (cu_seqlens, seq_lens, positions, column, owner, starts, columns)
        entry = self.graphs[key] = _Graph(graph=graph, layout=layout, tensors=static)
        while len(self.graphs) > self.max_graphs:
            self.graphs.popitem(last=False)
        logger.info_rank0("Captured prefill CUDA graph for complete chunks %s (%d rows)", key, rows)
        return entry
