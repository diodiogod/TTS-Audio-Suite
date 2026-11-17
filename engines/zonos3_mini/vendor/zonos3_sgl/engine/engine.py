from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import torch
from zonos3.models.config import ModelConfig

from zonos3_sgl.attention import create_attention_backend
from zonos3_sgl.core import (
    Context,
    TTSBatch,
    TTSReq,
    TTSSamplingParams,
    set_global_ctx,
)
from zonos3_sgl.distributed import destroy_distributed, set_tp_info
from zonos3_sgl.kvcache import create_kvcache
from zonos3_sgl.layers import set_rope_device
from zonos3_sgl.models import create_model, load_checkpoint_weight
from zonos3_sgl.utils.logger import init_logger
from zonos3_sgl.utils.misc import divide_even
from zonos3_sgl.utils.torch_utils import make_device_tensor, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .prefill_graph import PrefillGraphs

logger = init_logger(__name__)

# Single-request prefill graphs captured at startup: up to this many complete 64-row chunks (embedding prompts take
# 0-1, a default voice's reference prefix 4-5); other shapes are captured on first use.
PREFILL_PREWARM_CHUNKS = 5


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _get_frame_width(model_config) -> int:
    """Get frame width for multi-codebook models.

    Frame width = n_codebooks + (1 if text_vocab).
    """
    width = model_config.n_codebooks
    if model_config.text_vocab is not None:
        width += 1
    return width


class Engine:
    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        if self.model_config.attention_pattern == "gdn3_global1" and config.tp_info.size != 1:
            raise ValueError("This qualified GDN benchmark runtime requires TP1 (LM on one GPU, codec on the second)")
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        # Importing the model eagerly pulls in sgl_kernel, which creates a CUDA
        # context on the current device. The scheduler entry point binds this
        # process to its rank's GPU before that import, so an existing context is
        # expected; only one on the wrong device indicates a real ordering bug.
        if torch.cuda.is_initialized() and torch.cuda.current_device() != self.device.index:
            raise RuntimeError(
                f"CUDA was initialized on cuda:{torch.cuda.current_device()} before "
                f"the engine could bind TP rank {config.tp_info.rank} to {self.device}; "
                "set the device before importing model code."
            )
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # load model and determine number of pages
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config)

        # Log expected shapes before loading (for debugging dimension mismatches)
        logger.info_rank0(
            f"Model config: num_kv_heads={config.model_config.num_kv_heads}, head_dim={config.model_config.head_dim}"
        )
        expected_kv_dim = config.model_config.num_kv_heads * config.model_config.head_dim
        logger.info_rank0(f"Expected kv_dim per TP rank: {expected_kv_dim}")

        state_dict = self._load_weight_state_dict(config)
        grown = config.grown_vocabulary
        if grown is not None and ModelConfig.from_checkpoint_config(config.checkpoint_config).span_end_id is None:
            if config.tp_info.size != 1:
                raise ValueError("accent adapters do not support tensor parallelism")
            from zonos3.models.accent_format import grow_state_dict

            grow_state_dict(state_dict, grown, n_codebooks=self.model_config.n_codebooks)
            logger.info_rank0(
                "Vocabulary grown by the accent adapters: span_end_id %d, %d extra text rows",
                grown.span_end_id,
                grown.text_extra_tokens,
            )
        extra = config.text_extra_rows
        if extra is not None:
            if config.tp_info.size != 1:
                raise ValueError("--phoneme-rows does not support tensor parallelism")
            key = f"multi_embedder.embedders.{self.model_config.n_codebooks}.weight"
            state_dict[key] = extra.append(state_dict[key])
            logger.info_rank0(
                "Text rows appended from %s: %d (phoneme / start / pause tokens)", config.phoneme_rows, extra.count
            )
        self._check_speaker_lda_weights(state_dict)

        # Debug: Check for temp keys in state_dict
        if logger.isEnabledFor(logging.DEBUG):
            temp_keys = [k for k in state_dict.keys() if ".temp" in k]
            logger.debug("Found %d temp keys in state_dict: %s", len(temp_keys), temp_keys[:5])
            if temp_keys:
                first_temp = state_dict[temp_keys[0]]
                logger.debug(
                    "First temp tensor: shape=%s, dtype=%s, values=%s",
                    first_temp.shape,
                    first_temp.dtype,
                    first_temp.flatten()[:4].tolist(),
                )

        # Validate wkv weight shapes match config (catch n_kv_heads mismatch early)
        for key, tensor in state_dict.items():
            if ".wkv." in key and "weight" in key:
                # wkv.weight should be (2, kv_dim, hidden_size) or (2*kv_dim, hidden_size)
                if tensor.dim() == 3:
                    actual_kv_dim = tensor.shape[1]
                else:
                    actual_kv_dim = tensor.shape[0] // 2
                if actual_kv_dim != expected_kv_dim:
                    inferred_kv_heads = actual_kv_dim // config.model_config.head_dim
                    raise ValueError(
                        f"KV dimension mismatch for {key}: checkpoint has kv_dim={actual_kv_dim} "
                        f"(implying n_kv_heads={inferred_kv_heads}), but config expects "
                        f"kv_dim={expected_kv_dim} (n_kv_heads={config.model_config.num_kv_heads}). "
                        f"Please ensure your checkpoint's params.json has the correct n_kv_heads value "
                        f"matching the trained model weights."
                    )
                break  # Only need to check one layer

        self.model.load_state_dict(state_dict)
        self._uses_gdn = config.model_config.attention_pattern == "gdn3_global1"
        self._gdn_layers = []
        if self._uses_gdn:
            from zonos3_sgl.models.gdn import OriginalGDN

            for layer in self.model.layers.op_list:
                if isinstance(layer.attention, OriginalGDN):
                    layer.attention.allocate_cache(config.max_running_req + 1, self.device, self.dtype)
                    self._gdn_layers.append(layer.attention)
            logger.info_rank0("Original GDN: replicated mixer, independent FP32 states, fixed-window decode")
            self._check_head_batch_invariance()
        self._init_steering(config)
        self.accent_adapters = None
        if config.accent_adapters:
            # TTS Audio Suite Patch: accent adapters (zonos3_sgl.models.accent_adapters, expert_lora kernel) are not part
            # of this copy; the release checkpoint ships none.
            raise ValueError("accent adapters are not supported by the TTS Audio Suite copy of zonos3_sgl")

        # Debug: print attention temp values to verify they loaded correctly
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Model type: %s", type(self.model).__name__)
            logger.debug("Model has layers: %s", hasattr(self.model, "layers"))
            if hasattr(self.model, "layers"):
                logger.debug("layers type: %s", type(self.model.layers))
                if hasattr(self.model.layers, "op_list"):
                    logger.debug("num layers: %d", len(self.model.layers.op_list))
                    for i, layer in enumerate(self.model.layers.op_list[:2]):
                        logger.debug("Layer %d type: %s", i, type(layer).__name__)
                        if hasattr(layer, "attention"):
                            attn = layer.attention
                            logger.debug("Layer %d attention type: %s", i, type(attn).__name__)
                            logger.debug("Layer %d has_qk_norm: %s", i, getattr(attn, "has_qk_norm", "N/A"))
                            if hasattr(attn, "temp"):
                                temp = attn.temp
                                if temp is not None:
                                    logger.debug(
                                        "Layer %d attention.temp: shape=%s, values=%s",
                                        i,
                                        temp.shape,
                                        temp.flatten()[:8].tolist(),
                                    )
                                else:
                                    logger.debug("Layer %d attention.temp is None", i)

        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            device=self.device,
            dtype=self.dtype,
        )
        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        frame_width = _get_frame_width(self.model_config)
        self.dummy_req = TTSReq(
            input_ids=torch.zeros((1, frame_width), dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=TTSSamplingParams(),
            cache_handle=None,  # type: ignore
            n_codebooks=self.model_config.n_codebooks,
            audio_pad_id=self.model_config.audio_pad_id,
            stop_id=self.model_config.eoa_id,
        )
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)

        gdn_graph_bs = self._gdn_graph_batch_sizes(config, init_free_memory) if self._uses_gdn else None
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=gdn_graph_bs if self._uses_gdn else config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.audio_vocab_size,
            dummy_req=self.dummy_req,
            before_capture=self._gdn_before_capture if self._uses_gdn else None,
            after_capture=self._gdn_after_capture if self._uses_gdn else None,
            before_replay=self._gdn_before_replay if self._uses_gdn else None,
            probe_capacity=(config.align_probe_max_text, config.align_probe_max_masked, config.align_probe_max_keys),
            n_heads=self.model_config.num_qo_heads,
        )
        self.prefill_graphs = self._init_prefill_graphs(config) if self._uses_gdn else None

    # ------------------------------------------------------------------ prefill CUDA graphs
    def _init_prefill_graphs(self, config: EngineConfig) -> PrefillGraphs | None:
        """Prefill graphs for GDN checkpoints (engine/prefill_graph.py): prefills of up to
        ZONOS3_PREFILL_GRAPH_MAX_ROWS padded rows (1024), at most ZONOS3_PREFILL_GRAPHS graphs (16), single requests
        of up to PREFILL_PREWARM_CHUNKS complete chunks captured now; ZONOS3_PREFILL_CUDA_GRAPH=0 disables them (the
        eager prefill computes the same bits)."""
        if os.environ.get("ZONOS3_PREFILL_CUDA_GRAPH", "1").strip().lower() in {"0", "false", "no", "off"}:
            logger.info_rank0("Prefill CUDA graphs: disabled")
            return None
        graphs = PrefillGraphs(
            model=self.model,
            ctx=self.ctx,
            page_table=self.page_table,
            dummy_page=self.dummy_page,
            gdn_layers=self._gdn_layers,
            stream=self.stream,
            max_requests=config.max_running_req,
            max_rows=int(os.environ.get("ZONOS3_PREFILL_GRAPH_MAX_ROWS", "1024")),
            max_graphs=int(os.environ.get("ZONOS3_PREFILL_GRAPHS", "16")),
        )
        graphs.prewarm([(w,) for w in range(PREFILL_PREWARM_CHUNKS + 1)], dummy_slot=self.dummy_req.table_idx)
        return graphs

    # ------------------------------------------------------------------ GDN CUDA graphs
    def _gdn_graph_batch_sizes(self, config: EngineConfig, free_memory: int) -> list[int]:
        """Decode batch sizes captured for GDN checkpoints. The decode step is launch-bound (~1,000 kernels, ~3 ms of
        GPU work in ~35 ms), so graphs are on by default up to max_running_req; ZONOS3_GDN_CUDA_GRAPH=0 disables them
        (the eager fixed-window path computes the same bits)."""
        from zonos3_sgl.models.gdn import OriginalGDN

        from .graph import _determine_cuda_graph_bs

        if os.environ.get("ZONOS3_GDN_CUDA_GRAPH", "1").strip().lower() in {"0", "false", "no", "off"}:
            bs_list: list[int] = []
        elif config.cuda_graph_bs is not None:
            bs_list = list(config.cuda_graph_bs)
        else:
            max_bs = 160 if config.cuda_graph_max_bs is None else int(config.cuda_graph_max_bs)
            bs_list = _determine_cuda_graph_bs(None, min(max_bs, config.max_running_req), free_memory)
        bs_list = sorted({int(b) for b in bs_list if 0 < int(b) <= config.max_running_req})
        self._gdn_slots = torch.zeros((max(bs_list, default=1),), dtype=torch.long, device=self.device)
        tables = OriginalGDN.static_tables(self.device, bs_list)
        self._gdn_static_tensors = [t for pair in tables.values() for t in pair]
        for gdn in self._gdn_layers:
            gdn.enable_graph_decode(self._gdn_slots, tables)
        logger.info_rank0("GDN decode CUDA graphs: %s", bs_list if bs_list else "disabled")
        return bs_list

    def _gdn_fill_slots(self, batch: TTSBatch) -> None:
        slots = torch.tensor([r.table_idx for r in batch.padded_reqs], dtype=torch.long, pin_memory=True)
        self._gdn_slots[: len(slots)].copy_(slots, non_blocking=True)

    def _gdn_before_capture(self, batch: TTSBatch) -> None:
        for gdn in self._gdn_layers:
            gdn.set_static_mode(True)
        self._gdn_fill_slots(batch)

    def _gdn_after_capture(self) -> None:
        for gdn in self._gdn_layers:
            gdn.set_static_mode(False)

    def _gdn_before_replay(self, batch: TTSBatch) -> None:
        self._gdn_fill_slots(batch)

    # ------------------------------------------------------------------ hidden-space steering
    def _init_steering(self, config: EngineConfig) -> None:
        """Enable Full AttnRes bank steering when steering-bundle sets live in ``attnres_source``
        space. Each set steers its own bank source (one slot per distinct source; deltas sum when
        sets share one). The row buffer is static so the decode CUDA graphs can capture the adds;
        it is refilled from each decode batch's requests. Speaker-embedding (``raw``) sets change
        the request's embedding in the API server and need nothing here."""
        from zonos3.tts.directions import HIDDEN_SPACE, DirectionSet
        from zonos3.tts.steering import load_bundle

        self._steer_rows_dirty = False
        self.model.steer_sources = ()
        self.model.steer_rows = None
        steering: dict[str, DirectionSet] = {}
        bundle_path = getattr(config, "steer_bundle", None)
        if bundle_path:
            bundle = load_bundle(bundle_path)
            for set_name, directions in bundle.sets.items():
                if directions.space == HIDDEN_SPACE:
                    steering[f"bundle {bundle.name}.{set_name}"] = directions
        if not steering:
            return
        if self.model_config.residual_type != "full_attnres":
            raise ValueError(
                "attnres_source directions need a full_attnres checkpoint, "
                f"this one has residual_type={self.model_config.residual_type!r}"
            )
        n_sources = 2 * self.model_config.num_layers + 1
        for label, directions in steering.items():
            if directions.dim != self.model_config.hidden_size or not 0 <= directions.source_index < n_sources:
                raise ValueError(
                    f"{label}: attnres_source directions (dim {directions.dim}, source {directions.source_index}) "
                    f"do not fit hidden {self.model_config.hidden_size} with {n_sources} bank sources"
                )
        self.model.steer_sources = tuple(sorted({int(d.source_index) for d in steering.values()}))
        self.model.steer_rows = torch.zeros(
            (config.max_running_req + 1, len(self.model.steer_sources), self.model_config.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        logger.info_rank0(
            "Hidden-space steering on Full AttnRes bank sources: %s",
            ", ".join(f"{label} -> source {d.source_index}" for label, d in steering.items()),
        )

    def _fill_steer_rows(self, batch: TTSBatch) -> None:
        """Row i of the buffer = padded request i's delta at the frame it generates now (one
        hidden-size slot per steered source, flattened; zeros for none / dummies)."""
        rows = self.model.steer_rows
        assert rows is not None
        deltas = [r.steer_delta() for r in batch.padded_reqs]
        if not any(d is not None for d in deltas):
            if self._steer_rows_dirty:
                rows.zero_()
                self._steer_rows_dirty = False
            return
        staged = torch.zeros((len(deltas), rows.shape[1] * rows.shape[2]), dtype=torch.float32)
        for i, d in enumerate(deltas):
            if d is not None:
                staged[i] = d
        rows[: len(deltas)].copy_(
            staged.view(len(deltas), rows.shape[1], rows.shape[2]).to(dtype=rows.dtype).pin_memory(), non_blocking=True
        )
        if len(deltas) < rows.shape[0]:
            rows[len(deltas) :].zero_()
        self._steer_rows_dirty = True

    def _check_head_batch_invariance(self) -> None:
        """The exact-mode contract: a request's logits must not depend on how many rows share the head GEMM.
        Compares the padded head on 1..8 rows with the 1024-row product on this GPU / cuBLAS build and warns loudly
        if they differ (receipt: sgl_frozen_arch_20261012/checks/kernel-batch-probe-375k.json)."""
        head = self.model.multi_output
        generator = torch.Generator(device=self.device).manual_seed(20261012)
        x = torch.randn(1024, head.hidden_size, device=self.device, generator=generator).to(head.weight.dtype)
        reference = head.forward(x)
        worst = 0.0
        for rows in (1, 2, 3, 4, 5, 8):
            diff = (head.forward(x[:rows]).float() - reference[:rows].float()).abs().max().item()
            worst = max(worst, diff)
        if worst > 0:
            logger.warning_rank0(
                "LM head is not batch-invariant on this GPU (max |diff| %g between 1..8-row and 1024-row GEMMs); "
                "single-request logits will not match batched / training logits bit for bit. Set ZONOS3_HEAD_MIN_ROWS "
                "to a row count that reproduces the large-batch kernel.",
                worst,
            )
        else:
            logger.info_rank0("LM head batch invariance verified (padded to %d rows)", head.min_rows)

    def _init_communication(self, config: EngineConfig) -> None:
        # TTS Audio Suite Patch: one GPU, one process: no torch.distributed process group is created.
        if config.tp_info.size != 1:
            raise ValueError("zonos3_sgl in TTS Audio Suite does not support tensor parallelism")
        return None

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {k: torch.randn_like(v, device=self.device) for k, v in self.model.state_dict().items()}
        else:
            return {
                # Expert-selection biases are trained and applied in float32.
                # Keep that precision while casting the large model weights.
                k: v if k.endswith(".router.balancing_biases") else v.to(self.dtype)
                for k, v in load_checkpoint_weight(config.model_path, self.device).items()
            }

    def _check_speaker_lda_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        lda_keys = [key for key in state_dict if key.startswith("speaker_lda_projection.")]
        if getattr(self.model, "speaker_lda_projection", None) is None:
            for key in lda_keys:
                state_dict.pop(key)
            if lda_keys:
                logger.warning_rank0(
                    "Dropped %d speaker_lda_projection weights because speaker LDA is not enabled.",
                    len(lda_keys),
                )
            return

        missing = {
            "speaker_lda_projection.weight",
            "speaker_lda_projection.bias",
        }.difference(lda_keys)
        if missing:
            raise ValueError(
                "speaker_lda_dim is configured but the checkpoint is missing "
                f"{sorted(missing)}. Use a checkpoint with the LDA projection merged in."
            )

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        return free_memory, free_memory

    def forward_batch_tts(self, batch: TTSBatch) -> torch.Tensor:
        """Forward pass for TTS batch, returning multi-codebook logits.

        Args:
            batch: TTS batch with 2D input tokens

        Returns:
            Logits tensor of shape (batch_size, n_codebooks, vocab_size)
        """
        assert torch.cuda.current_stream() == self.stream
        if batch.is_prefill and self.prefill_graphs is not None:
            key = self.prefill_graphs.key(batch)
            if key is not None:
                return self.prefill_graphs.run(batch, key)
        if batch.is_decode and self.model.steer_rows is not None:
            self._fill_steer_rows(batch)
        with self.ctx.forward_batch(batch):
            # For TTS, the model outputs multi-codebook logits directly
            # The zonos3 model has a MultiParallelLMHead that outputs
            # (batch, seq, n_codebooks, vocab_size)
            use_graph = self.graph_runner.can_use_cuda_graph(batch)
            if batch.is_decode:
                from zonos3_sgl.tts import (
                    align_probe,  # lazy: zonos3_sgl.tts imports the scheduler, which imports this module
                )

                align_probe.prepare(
                    batch,
                    self.graph_runner.probe_static if use_graph else None,
                    self.model_config.num_qo_heads,
                    self.device,
                )
            if use_graph:
                logits = self.graph_runner.replay(batch)
                for gdn in self._gdn_layers:
                    gdn.note_graph_forward(batch.reqs)
            else:
                logits = self.model.forward()

        # logits shape: (batch, n_codebooks, vocab) for decode
        # or (total_tokens, n_codebooks, vocab) for prefill
        # We only care about the last token per sequence
        if batch.is_decode:
            return logits[: batch.size]
        else:
            # For prefill, get the LAST token logits for each sequence
            # logits shape: (total_tokens, n_codebooks, vocab)
            # Need to extract logits at the last position of each sequence
            last_indices = []
            cumsum = 0
            for req in batch.reqs:
                cumsum += req.extend_len  # extend_len = device_len - cached_len
                last_indices.append(cumsum - 1)
            return logits[make_device_tensor(last_indices, torch.long, logits.device)]

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        destroy_distributed()
