from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from zonos3.utils.hf import cached_load_checkpoint_config

from zonos3_sgl.distributed import DistributedInfo

if TYPE_CHECKING:
    from zonos3.models.config import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "fused_moe"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # Directory of accent adapters (<accent>.pt) served per request; see
    # zonos3_sgl.models.accent_adapters. None serves the base model only.
    accent_adapters: str | None = None
    # zonos-text-extra/1 rows (phoneme / start / pause tokens; zonos3.models.text_extra; a path or hf://) appended to
    # the text table at load, after the vocabulary a speech-edit adapter grows. None serves the checkpoint's text rows.
    phoneme_rows: str | None = None
    # Alignment probes (zonos3_sgl.tts.align_probe) inside the decode CUDA graphs: text keys read per request (0 = graphs
    # carry no probe and any decode batch with a probe runs eager), masked requests per batch, keys per masked request.
    align_probe_max_text: int = 0
    align_probe_max_masked: int = 8
    align_probe_max_keys: int = 12288

    @cached_property
    def checkpoint_config(self):
        return cached_load_checkpoint_config(self.model_path)

    @cached_property
    def grown_vocabulary(self):
        """The vocabulary the ``zonos-accent-lora/3`` adapters of ``accent_adapters`` grow the model to (None: as is)."""
        from zonos3.models.accent_format import grown_vocabulary

        return grown_vocabulary(self.accent_adapters)

    @cached_property
    def text_extra_rows(self):
        """The ``phoneme_rows`` export (zonos3.models.text_extra.TextExtraRows), or None."""
        from zonos3.models.text_extra import TextExtraRows

        return TextExtraRows.load(self.phoneme_rows) if self.phoneme_rows else None

    @cached_property
    def model_config(self) -> ModelConfig:
        from dataclasses import replace

        from zonos3.models.config import ModelConfig

        config = ModelConfig.from_checkpoint_config(self.checkpoint_config)
        grown = self.grown_vocabulary
        if grown is not None and (config.span_end_id, config.text_extra_tokens) != (
            grown.span_end_id,
            grown.text_extra_tokens,
        ):
            if config.span_end_id is not None or config.text_extra_tokens:
                raise ValueError(
                    f"the accent adapters grow the vocabulary (span_end_id {grown.span_end_id}, "
                    f"{grown.text_extra_tokens} extra text rows) of a checkpoint that already has span_end_id "
                    f"{config.span_end_id} and {config.text_extra_tokens} extra text rows"
                )
            config = replace(
                config,
                span_end_id=grown.span_end_id,
                text_extra_tokens=grown.text_extra_tokens,
                vocab_size=config.vocab_size + config.n_codebooks + grown.text_extra_tokens,
            )
        extra = self.text_extra_rows
        if extra is not None:
            config = replace(
                config,
                text_extra_tokens=config.text_extra_tokens + extra.count,
                vocab_size=config.vocab_size + extra.count,
            )
        return config

    @cached_property
    def text_tokenizer_spec(self) -> dict | None:
        """The served text tokenizer spec: the checkpoint's (zonos3.tts.superbpe.load_spec) with the phonemes / start /
        pauses of ``phoneme_rows``, their ids after the extra rows the model already has."""
        from zonos3.tts.superbpe import load_spec

        spec = load_spec(self.model_path)
        extra = self.text_extra_rows
        if extra is None:
            return spec
        existing = self.model_config.text_extra_tokens - extra.count
        return extra.spec(spec, text_vocab=int(self.model_config.text_vocab), existing=existing)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        import os

        # Several offline engines on one node (evaluations, parity checks) must not share the rendezvous port.
        return os.environ.get("ZONOS3_DISTRIBUTED_ADDR", "tcp://127.0.0.1:23333")
