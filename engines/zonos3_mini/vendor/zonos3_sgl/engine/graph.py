from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Dict, List

import torch

from zonos3_sgl.core import TTSBatch, TTSReq, get_global_ctx
from zonos3_sgl.utils.logger import init_logger

if TYPE_CHECKING:
    from zonos3_sgl.attention import BaseAttnBackend
    from zonos3_sgl.models import Zonos3ForCausalLM

logger = init_logger(__name__)


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: Zonos3ForCausalLM,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: TTSReq,
        before_capture=None,
        after_capture=None,
        before_replay=None,
        probe_capacity: tuple[int, int, int] = (0, 0, 0),
        n_heads: int = 0,
    ) -> None:
        # Optional model-side hooks (GDN request-state buffers): called with the padded dummy batch before each
        # capture, once after all captures, and with the real padded batch before each replay.
        self.before_capture = before_capture
        self.after_capture = after_capture
        self.before_replay = before_replay
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs, default=0)
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.probe_capacity = tuple(int(x) for x in probe_capacity)
        self.n_heads = int(n_heads)
        self.graph_map = self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: Zonos3ForCausalLM):
        graph_map: Dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}  # (batch size, masked-heads variant)
        self.probe_static = None
        if self.max_graph_bs == 0:
            logger.info_rank0("CUDA graph is disabled.")
            return graph_map
        max_text, max_masked, max_keys = self.probe_capacity
        if max_text > 0:
            from zonos3_sgl.tts.align_probe import ProbeStatic, attention_layers

            max_keys = min(int(max_keys), max_seq_len)  # masked requests read their keys from the captured page table
            self.probe_static = ProbeStatic(
                self.max_graph_bs, max_text, max_masked, max_keys, len(attention_layers()), self.n_heads, self.device
            )
            logger.info_rank0(
                f"Decode graphs carry the alignment-probe readout ({max_text} text keys) and up to {max_masked} "
                f"masked requests of <= {max_keys} keys"
            )

        self.logits = torch.empty(
            (self.max_graph_bs, self.dummy_req.n_codebooks, vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        # Accent adapters: per-row slots and strengths are graph inputs, like
        # input_ids. Padding rows stay on slot 0 (base) at strength 0.
        self.accent_slots = self.accent_scales = None
        if model.__dict__.get("_accents") is not None:
            self.accent_slots = torch.zeros(self.max_graph_bs, dtype=torch.int32, device=self.device)
            self.accent_scales = torch.zeros(self.max_graph_bs, dtype=torch.float32, device=self.device)
        frame_width = self.dummy_req.frame_width
        self.attn_backend.init_capture_graph(
            max_seq_len=max_seq_len, bs_list=self.graph_bs_list, frame_width=frame_width
        )

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # With alignment probes, each batch size gets a second graph that also recomputes masked heads; a batch replays
        # it only when it has masked requests (align_probe.prepare wrote their slots for this step).
        variants = (False, True) if self.probe_static is not None and self.probe_static.max_masked > 0 else (False,)
        pool = None
        for bs in sorted(self.graph_bs_list, reverse=True):
            for masked in variants:
                graph = torch.cuda.CUDAGraph()
                batch = TTSBatch(reqs=[self.dummy_req] * bs, phase="decode")
                batch.padded_reqs = batch.reqs
                if self.probe_static is not None:
                    batch.probe = self.probe_static.inputs(bs, [], masked)
                self.attn_backend.prepare_for_capture(batch)
                if self.accent_slots is not None:
                    batch.accent_slots = self.accent_slots[:bs]
                    batch.accent_scales = self.accent_scales[:bs]
                if self.before_capture is not None:
                    self.before_capture(batch)
                with get_global_ctx().forward_batch(batch):
                    self.logits[:bs] = model.forward()
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        self.logits[:bs] = model.forward()
                if pool is None:
                    pool = graph.pool()
                graph_map[(bs, masked)] = graph

        if self.after_capture is not None:
            self.after_capture()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")
        return graph_map

    def can_use_cuda_graph(self, batch: TTSBatch) -> bool:
        if not batch.is_decode or batch.size > self.max_graph_bs:
            return False
        from zonos3_sgl.tts.align_probe import graph_compatible  # lazy: zonos3_sgl.tts imports the scheduler

        return graph_compatible(batch, self.probe_static)

    def replay(self, batch: TTSBatch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        g = self.graph_map[(batch.padded_size, batch.probe is not None and batch.probe.masked is not None)]
        self.attn_backend.prepare_for_replay(batch)
        if self.accent_slots is not None:
            size = batch.padded_size
            self.accent_slots[:size].copy_(batch.accent_slots)
            self.accent_scales[:size].copy_(batch.accent_scales)
        if self.before_replay is not None:
            self.before_replay(batch)
        g.replay()
        return self.logits[: batch.size]

    def pad_batch(self, batch: TTSBatch) -> int:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size) if self.can_use_cuda_graph(batch) else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()
