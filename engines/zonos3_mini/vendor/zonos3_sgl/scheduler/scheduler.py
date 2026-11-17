"""TTS Scheduler for multi-codebook audio generation."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from zonos3.speaker_cfg import guided_logits, speaker_cfg_twin
from zonos3.tts.delay import apply_cold_start_delay

from zonos3_sgl.core import TTSBatch, TTSReq
from zonos3_sgl.env import ENV
from zonos3_sgl.message import (
    BaseTTSBackendMsg,
    BatchTTSBackendMsg,
    BatchTTSTokenizerMsg,
    ExitMsg,
    TTSDetokenizeMsg,
    TTSUserMsg,
)
from zonos3_sgl.utils.logger import init_logger
from zonos3_sgl.utils.torch_utils import make_device_tensor, to_device

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import TTSDecodeManager
from .io import SchedulerIOMixin
from .table import TTSTableManager

if TYPE_CHECKING:
    from zonos3.tts.laci import LaciSession

    from zonos3_sgl.engine.sample import TTSBatchSamplingArgs

logger = init_logger(__name__)


class TTSForwardOutput(NamedTuple):
    """Output from TTS forward pass."""

    next_tokens_gpu: torch.Tensor  # (batch, frame_width)
    next_tokens_cpu: torch.Tensor  # (batch, frame_width)
    copy_done: torch.cuda.Event
    probe_cpu: torch.Tensor | None = None  # (batch, 2) alignment-probe readout (zonos3_sgl.tts.align_probe), or None


class TTSForwardInput(NamedTuple):
    """Input for TTS forward pass."""

    batch: TTSBatch
    sample_args: TTSBatchSamplingArgs
    load_indices: torch.Tensor
    write_indices: torch.Tensor


TTSForwardData: TypeAlias = "Tuple[TTSForwardInput, TTSForwardOutput]"


@dataclass(eq=False)
class _LaciRequest:
    """A request under LACI: its session and the request as received (continuations are rebuilt from it)."""

    session: "LaciSession"
    template: TTSUserMsg
    context_sent: bool = False  # the vocoder's decode context / prefill rows went out with the first emitted row


def _make_3d_indices(table_3d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """Return 1D indices for a 3D token pool.

    Args:
        table_3d: 3D tensor (max_reqs, max_seq_len, frame_width)
        ranges: List of (req_idx, begin, end) tuples

    Returns:
        1D tensor of indices into the flattened token pool.
        These indices are designed to be divided by frame_width to get
        row indices into flat_pool.view(-1, frame_width).
    """
    assert table_3d.dim() == 3 and table_3d.is_contiguous()
    req_stride, seq_stride, _ = table_3d.stride()
    # For shape (max_reqs, max_seq_len, frame_width):
    #   req_stride = max_seq_len * frame_width
    #   seq_stride = frame_width
    # We compute: entry * req_stride + pos * seq_stride
    # When divided by frame_width, this gives: entry * max_seq_len + pos
    needed_size = sum(end - begin for _, begin, end in ranges)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        base = entry * req_stride
        for i in range(length):
            indices_host[offset + i] = base + (begin + i) * seq_stride
        offset += length
    return indices_host.to(table_3d.device, non_blocking=True)


def _make_speaker_slot(
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
) -> torch.Tensor:
    """Create a reserved frame for speaker embedding injection at position 0."""
    assert input_ids.dim() == 2
    slot = torch.zeros(
        (1, input_ids.shape[-1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    slot[:, :audio_cols] = int(audio_pad_id)
    if input_ids.shape[-1] > n_codebooks:
        text_pad = int(text_vocab) if int(text_vocab) > 0 else int(audio_pad_id)
        slot[:, n_codebooks] = text_pad
    return slot


def _has_reserved_speaker_slot(
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
) -> bool:
    """Check whether the first frame already matches the reserved speaker slot."""
    if input_ids.shape[0] == 0 or input_ids.shape[-1] <= n_codebooks:
        return False
    first = input_ids[0]
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    audio_ok = bool(torch.all(first[:audio_cols] == int(audio_pad_id)))
    text_expected = int(text_vocab) if int(text_vocab) > 0 else int(audio_pad_id)
    text_ok = int(first[n_codebooks].item()) == text_expected
    return audio_ok and text_ok


def _make_marker_slot(
    input_ids: torch.Tensor,
    *,
    n_codebooks: int,
    audio_pad_id: int,
    text_token: int,
) -> torch.Tensor:
    """Create a conditioning marker frame: audio padding plus one text token."""
    assert input_ids.dim() == 2
    slot = torch.zeros(
        (1, input_ids.shape[-1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    slot[:, :audio_cols] = int(audio_pad_id)
    if input_ids.shape[-1] > n_codebooks:
        slot[:, n_codebooks] = int(text_token)
    return slot


def _frame_text_token(
    input_ids: torch.Tensor,
    frame_idx: int,
    *,
    n_codebooks: int,
    audio_pad_id: int,
) -> int | None:
    """Return the text token of an audio-padded frame, or None if not a marker."""
    if input_ids.shape[0] <= frame_idx or input_ids.shape[-1] <= n_codebooks:
        return None
    frame = input_ids[frame_idx]
    audio_cols = min(n_codebooks, input_ids.shape[-1])
    if not bool(torch.all(frame[:audio_cols] == int(audio_pad_id))):
        return None
    return int(frame[n_codebooks].item())


def _apply_cold_start_delay_mask(
    next_tokens: torch.Tensor,
    batch: TTSBatch,
    *,
    n_codebooks: int,
    audio_pad_id: int,
) -> None:
    """zonos3.tts.delay.apply_cold_start_delay on every row of the batch."""
    for row, req in enumerate(batch.reqs):
        apply_cold_start_delay(
            next_tokens[row],
            int(req.total_generated),
            n_codebooks=n_codebooks,
            pad_id=int(audio_pad_id),
            has_audio_prefix=req.has_audio_prefix,
        )


def _speaking_rate_buckets(model_config, condition_unknowns: Tuple[int, ...]):
    """The checkpoint's speaking-rate condition row as a LACI RateBuckets (its own table's bucket count), or None."""
    from zonos3.tts.laci import RateBuckets  # local import: zonos3.tts imports this module

    names = list(getattr(model_config, "condition_names", None) or [])
    edges = (getattr(model_config, "condition_edges", None) or {}).get("speaking_rate")
    if not condition_unknowns or "speaking_rate" not in names or not edges:
        return None
    index = names.index("speaking_rate")
    return RateBuckets(index, tuple(float(e) for e in edges), int(condition_unknowns[index]))


class TTSScheduler(SchedulerIOMixin):
    """Scheduler for TTS multi-codebook generation."""

    def __init__(self, config: SchedulerConfig):
        from zonos3_sgl.engine import Engine
        from zonos3_sgl.engine.sample import TTSSampler

        self.engine = Engine(config)
        super().__init__(config, self.engine.tp_cpu_group)

        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # TTS-specific configuration - read from model config
        model_config = self.engine.model_config
        self.n_codebooks = model_config.n_codebooks
        self.codebook_size = model_config.codebook_size
        self.codebook_sizes = tuple(model_config.codebook_sizes or ())
        self.eoa_id = model_config.eoa_id
        if config.eoa_policy not in ("any", "codebook0"):
            raise ValueError(f"eoa_policy must be 'any' or 'codebook0', got {config.eoa_policy!r}")
        self.eoa_policy = config.eoa_policy
        self.audio_pad_id = model_config.audio_pad_id
        self.span_end_id = model_config.span_end_id
        self.text_vocab = model_config.text_vocab if model_config.text_vocab is not None else 0
        self.speaker_enabled = bool(getattr(model_config, "speaker_enabled", False))
        # GDN-family checkpoints decode eagerly; sample them from float32 logits like the training generator.
        from zonos3.tts.sampler import sample_logits_in_fp32  # local import: zonos3.tts imports this module

        self.sample_fp32_logits = sample_logits_in_fp32(model_config)
        self.speaker_embedding_dim = int(getattr(model_config, "speaker_embedding_dim", 0))
        self.speaker_embedding_codec = getattr(model_config, "speaker_embedding_codec", None)
        # Condition rows (cond3 finetunes): one table per condition, its id condition_bucket_counts[i] = unknown.
        self.condition_count = (
            int(getattr(model_config, "condition_count", 0) or 0)
            if getattr(model_config, "condition_tokens_enabled", False)
            else 0
        )
        self.condition_unknowns = (
            tuple(int(n) for n in model_config.condition_bucket_counts) if self.condition_count else ()
        )
        self.speaker_background_token_enabled = bool(getattr(model_config, "speaker_background_token_enabled", False))
        self.accurate_mode_token_enabled = bool(getattr(model_config, "accurate_mode_token_enabled", False))
        self.speaker_background_num_buckets = 2 if self.speaker_background_token_enabled else 0
        self.accurate_mode_num_buckets = (
            1 if self.accurate_mode_token_enabled and self.speaker_background_num_buckets > 0 else 0
        )
        self.speaking_rate_num_buckets = int(getattr(model_config, "speaking_rate_num_buckets", 0))
        quality_buckets = getattr(model_config, "quality_buckets", None) or {}
        quality_features = tuple(getattr(model_config, "quality_features", ()) or ())
        if not quality_features:
            quality_features = tuple(quality_buckets.keys())
        self.quality_features = quality_features
        self.quality_bucket_counts = [len(quality_buckets.get(feature, ())) for feature in quality_features]
        # Frame width includes n_codebooks audio codes and one text token.
        self.frame_width = self.n_codebooks + 1

        # TTS sampler
        self.tts_sampler = TTSSampler(
            device=self.device,
            n_codebooks=self.n_codebooks,
            codebook_size=self.codebook_size,
            codebook_sizes=self.codebook_sizes,
            text_vocab=self.text_vocab,
            audio_pad_id=self.audio_pad_id,
            end_token_ids=(self.eoa_id,) if self.span_end_id is None else (self.eoa_id, self.span_end_id),
        )

        # Initialize TTS-specific managers
        self.table_manager = TTSTableManager(config.max_running_req, self.engine.page_table, self.frame_width)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = TTSDecodeManager()

        # Waiting queue for prefill
        self.waiting_reqs: List[TTSUserMsg] = []

        self.tp_info = config.tp_info
        self.finished_reqs: Set[TTSReq] = set()
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

        # Metrics tracking for periodic throughput logging
        self._stats_start = time.perf_counter()
        self._stats_batches = 0
        self._stats_tokens = 0

        # LACI (zonos3.tts.laci): alignment-head specs by language key, the live requests by uid.
        from zonos3.tts.laci import LaciConfig, LaciHeads  # local import: zonos3.tts imports this module

        self.laci_config = LaciConfig()
        self.laci_heads = {key: LaciHeads.load(path) for key, path in config.laci_heads.items()}
        self.laci_requests: Dict[int, _LaciRequest] = {}
        # Duration targets (TTSUserMsg.laci_target_frames) regenerate a missed take in another speaking-rate bucket: the
        # checkpoint's speaking-rate condition row, and a cache that never serves a prompt prefix with stale condition
        # rows (the continuation re-prefills its prompt with the new ids).
        self.laci_rates = _speaking_rate_buckets(model_config, self.condition_unknowns)
        self.laci_rate_switch = config.cache_type == "naive"

    def _process_last_data(
        self,
        last_data: TTSForwardData | None,
        ongoing_data: TTSForwardData | None,
    ) -> None:
        if last_data is None:
            return

        batch, output = last_data[0].batch, last_data[1]
        next_tokens_cpu, copy_done = output.next_tokens_cpu, output.copy_done
        copy_done.synchronize()
        if output.probe_cpu is not None:
            from zonos3_sgl.tts.align_probe import deliver  # lazy: zonos3_sgl.tts imports this module

            # observers see this step before EOS/length bookkeeping, so a stop finishes the request at this row
            deliver(batch, output.probe_cpu, self.finished_reqs)
        reply = BatchTTSTokenizerMsg(data=[])
        logger.debug(
            "Processing batch: size=%d, padded_size=%d, next_tokens shape=%s",
            batch.size,
            batch.padded_size,
            next_tokens_cpu.shape,
        )

        max_seq_len = self.engine.max_seq_len
        for i, req in enumerate(batch.reqs):
            if req in self.finished_reqs:
                continue

            # Get audio codes for this frame
            next_token = next_tokens_cpu[i]  # (frame_width,)
            audio_codes = next_token[: self.n_codebooks].tolist()

            # Append to host storage
            req.append_host(next_token)

            # Unconditional CFG twins are internal: no output, no EOS handling.
            # Their lifecycle is tied to the conditional req below.
            if req.is_cfg_uncond:
                continue

            # Log progress
            generated = req.total_generated
            if generated <= 5:
                logger.debug("TTS %d frame %d: audio_codes=%s", req.uid, generated, audio_codes)

            # Check for EOS
            finished = req.remain_len <= 0
            if req.check_eos(audio_codes):
                finished = True
                logger.debug(
                    "TTS %d EOS detected at frame %d, eos_frame=%d",
                    req.uid,
                    generated,
                    req.eos_frame,
                )

            if req.device_len >= max_seq_len - 1:
                finished = True
                logger.warning_rank0(f"TTS request {req.uid} reached {max_seq_len = }, dropped.")

            laci = self.laci_requests.get(req.uid)
            if laci is not None:
                # LACI: the session owns the stream; only rows no later rollback can change go downstream
                laci.session.append(next_token, req.eos_frame if req.eos_frame >= 0 else None)
                reply.data.extend(self._laci_rows(req, laci, finished))
            else:
                # Send detokenize message with audio codes
                reply.data.append(
                    TTSDetokenizeMsg(
                        uid=req.uid,
                        audio_codes=audio_codes,
                        finished=finished,
                        eos_frame=req.eos_frame if req.eos_frame >= 0 else None,
                        stream_decode=getattr(req.sampling_params, "stream_decode", None),
                        stream_codebooks=req.sampling_params.stream_codebooks,
                        stream_codebooks_scope=req.sampling_params.stream_codebooks_scope,
                        decoder=req.sampling_params.decoder,
                        # the vocoder's decode context / prefilled rows ride on the request's first row (2026-10-17)
                        decode_context_codes=None if req.context_sent else req.decode_context_codes,
                        prefill_rows=None if req.context_sent else req.prefill_rows,
                        end_token=req.sampling_params.end_token,
                    )
                )
                if not req.context_sent and (req.decode_context_codes is not None or req.prefill_rows is not None):
                    logger.info(
                        "TTS %d: decode context sent with generated row %d (context %s, prefill %s)",
                        req.uid,
                        generated,
                        None if req.decode_context_codes is None else tuple(req.decode_context_codes.shape),
                        None if req.prefill_rows is None else tuple(req.prefill_rows.shape),
                    )
                req.context_sent = True

            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                # Finish the unconditional twin alongside its conditional req.
                if req.cfg_twin is not None:
                    self.finished_reqs.add(req.cfg_twin)
                    self.decode_manager.remove_req(req.cfg_twin)
                if os.environ.get("LONG_TTS_EVAL") == "1":
                    logger.info(
                        "LONG_FINISH uid=%s generated=%s eos_frame=%s context_frames=%s output_limit=%s",
                        req.uid,
                        generated,
                        req.eos_frame,
                        req.device_len,
                        req.sampling_params.max_tokens,
                    )
                logger.debug("TTS %d finished: %d frames generated", req.uid, generated)
                dump_dir = os.environ.get("ZONOS3_DUMP_TOKENS")
                if dump_dir:
                    import numpy as np

                    Path(dump_dir).mkdir(parents=True, exist_ok=True)
                    rows = self.token_pool[req.table_idx, : req.device_len].cpu().numpy()
                    np.save(Path(dump_dir) / f"uid{req.uid}.npy", rows)
                    (Path(dump_dir) / f"uid{req.uid}.json").write_text(
                        json.dumps(
                            {
                                "uid": req.uid,
                                "prompt_rows": int(len(req.input_ids)),
                                "device_len": int(req.device_len),
                                "generated": int(generated),
                                "eos_frame": req.eos_frame,
                                "is_cfg_uncond": bool(req.is_cfg_uncond),
                            }
                        )
                    )

        # Free resources for finished but not ongoing reqs
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            # Free the allocated cache slots (cached_len represents actually allocated positions)
            slots = self.page_table[req.table_idx, : req.cached_len]
            self.cache_manager.free_slots(slots)
            self.cache_manager.free_handle(req.cache_handle)
            self.table_manager.free(req.table_idx)

        self.finished_reqs.intersection_update(ongoing_reqs)
        self.send_result(reply)

    # ------------------------------------------------------------------ LACI (zonos3.tts.laci)
    def check_laci(self, key: str, sampling_params) -> None:
        """LACI's constraints for a request (raises ValueError): a configured head spec, no CFG twin (the probe reads
        and masks the conditional request alone), an utterance end (an edit span ends on span-end, not on the text's
        EOS the heads read to), the FA attention backend (the probe reads its KV layout), TP 1 (the probe sees every
        head)."""
        from zonos3_sgl.attention.fa import FlashAttentionBackend

        if key not in self.laci_heads:
            raise ValueError(f"no LACI alignment heads for {key!r} (configured: {sorted(self.laci_heads) or 'none'})")
        if (
            float(sampling_params.speaker_cfg_scale) != 1.0
            or float(sampling_params.controls_cfg_scale) != 1.0
        ):
            raise ValueError("LACI needs speaker_cfg_scale = controls_cfg_scale = 1")
        if sampling_params.end_token != "utterance":
            raise ValueError("LACI needs end_token = 'utterance' (no edit spans)")
        if not isinstance(self.engine.attn_backend, FlashAttentionBackend):
            raise ValueError("LACI needs the 'fa' attention backend")
        if self.tp_info.size != 1:
            raise ValueError("LACI needs tensor parallel size 1")

    def _laci_probe(self, msg: TTSUserMsg):
        """The alignment probe of a LACI request's next attempt (a new session for a new request)."""
        from zonos3.tts.laci import DurationTarget, LaciSession, RateBuckets

        laci = self.laci_requests.get(msg.uid)
        if laci is None:
            self.check_laci(msg.laci, msg.sampling_params)
            frames = getattr(msg, "laci_target_frames", None)
            duration = None if frames is None else DurationTarget(float(frames), adjust_rate=bool(msg.laci_adjust_rate))
            if duration is not None and duration.adjust_rate and not self.laci_rate_switch:
                raise ValueError("a LACI duration target with adjust_rate needs cache_type='naive'")
            prefill = msg.prefill_rows
            rates, bucket = self.laci_rates, None
            if msg.laci_rate_edges is not None:  # a take on an extended speaking-rate scale (steered buckets)
                if rates is None:
                    raise ValueError("an extended speaking-rate scale needs the checkpoint's speaking-rate condition")
                rates = RateBuckets(
                    rates.index, tuple(float(e) for e in msg.laci_rate_edges), rates.n_buckets, int(msg.laci_rate_first)
                )
                bucket = int(msg.laci_rate_bucket)
                if not rates.first <= bucket <= rates.last:
                    raise ValueError(
                        f"speaking-rate bucket {bucket} is outside the scale [{rates.first}, {rates.last}]"
                    )
            session = LaciSession(
                heads=self.laci_heads[msg.laci],
                cfg=self.laci_config,
                prefill_rows=prefill if prefill is not None else torch.zeros((0, self.frame_width), dtype=torch.int32),
                seed=msg.sampling_params.seed,
                max_tokens=int(msg.sampling_params.max_tokens),
                n_codebooks=self.n_codebooks,
                hold=msg.sampling_params.stream_decode != "prefix",
                duration=duration,
                rates=rates,
                condition_values=None if msg.condition_values is None else [int(v) for v in msg.condition_values],
                bucket=bucket,
                rate_deltas=None if msg.laci_rate_edges is None else list(msg.laci_rate_deltas),
                follower=self._span_follower(msg),
                short_text=msg.laci_short_text,
            )
            laci = self.laci_requests[msg.uid] = _LaciRequest(session, msg)
        from zonos3_sgl.tts.align_probe import laci_probe

        return laci_probe(laci.session)

    def _laci_stop_bias(self, batch) -> list[float] | None:
        """Per request of ``batch``: the stop-id logit bias of its LACI session (LaciSession.stop_boost), 0 for others;
        None when no request has one. With overlap scheduling the session may lag the sampled row by one frame (the
        boost then starts a row late)."""
        if not self.laci_requests:
            return None
        bias = []
        for req in batch.reqs:
            laci = self.laci_requests.get(req.uid)
            bias.append(0.0 if laci is None else laci.session.stop_boost())
        return bias if any(bias) else None

    def _span_follower(self, msg: TTSUserMsg, origin: int = 0):
        """The span gate of a request whose control spans follow the reading position (zonos3.tts.span_follow), or
        None: no such spans, or no alignment heads for its language (then the character-rate schedule places them).
        The probe needs what LACI's does (the FA backend, TP 1: --laci-heads enforces both); unlike LACI it only
        reads the conditional request, so CFG twins are fine."""
        from zonos3.tts.span_follow import FollowConfig, SpanFollower

        key, schedule = getattr(msg, "steer_follow", None), getattr(msg, "steer_schedule", None)
        if key is None or schedule is None:
            return None
        if key not in self.laci_heads:
            logger.warning(
                "TTS %s: no alignment heads for %r, its control spans keep the character-rate schedule", msg.uid, key
            )
            return None
        lead, lag = (
            msg.steer_follow_gate if msg.steer_follow_gate is not None else (FollowConfig.lead, FollowConfig.lag)
        )
        return SpanFollower(
            msg.steer_follow_tokens, schedule[:, 3:], FollowConfig(lead=float(lead), lag=float(lag)), origin=origin
        )

    def _laci_rows(self, req: TTSReq, laci: _LaciRequest, finished: bool) -> list[TTSDetokenizeMsg]:
        """Detokenize messages for the rows of a LACI request that became final; on finish, either queue the rollback's
        continuation (ahead of new requests) or flush the rest with the final message."""
        session = laci.session
        final = False
        if finished:
            continuation = session.finish_attempt()
            if continuation is not None:
                t = laci.template
                prompt = t.input_ids[: len(t.input_ids) - session.n_prefill]
                self.waiting_reqs.insert(
                    0,
                    replace(
                        t,
                        input_ids=torch.cat((prompt, continuation.rows.to(prompt.dtype)), dim=0),
                        sampling_params=replace(
                            t.sampling_params, seed=continuation.seed, max_tokens=continuation.max_tokens
                        ),
                        history_rows=continuation.history_rows,
                        condition_values=(
                            continuation.condition_values
                            if continuation.condition_values is not None
                            else t.condition_values
                        ),
                        steering_delta=(continuation.steering_delta if continuation.swap_delta else t.steering_delta),
                    ),
                )
                failure = session.failures[-1]
                logger.info(
                    "LACI uid=%s: %s (bin %s) at row %s, rolling back to row %s, retry %d/%d",
                    req.uid,
                    failure["kind"],
                    failure["bin"],
                    failure["detect_row"],
                    failure["rollback_row"],
                    session.attempts,
                    session.cfg.max_retries,
                )
            else:
                final = True
        rows = session.drain(final=final)
        record = session.record() if final else None
        if final and not rows:
            raise AssertionError(f"LACI uid={req.uid}: no row left for the final message")  # commit_row holds one back
        msgs = []
        for k, row in enumerate(rows):
            last = final and k == len(rows) - 1
            first = not laci.context_sent
            msgs.append(
                TTSDetokenizeMsg(
                    uid=req.uid,
                    audio_codes=row[: self.n_codebooks].tolist(),
                    finished=last,
                    eos_frame=session.eos_frame() if last else None,
                    stream_decode=getattr(req.sampling_params, "stream_decode", None),
                    decoder=req.sampling_params.decoder,
                    decode_context_codes=laci.template.decode_context_codes if first else None,
                    prefill_rows=laci.template.prefill_rows if first else None,
                    laci=record if last else None,
                    end_token=req.sampling_params.end_token,
                )
            )
            laci.context_sent = True
        if final:
            del self.laci_requests[req.uid]
            logger.info(
                "LACI uid=%s: end=%s attempts=%d rows=%d (generated %d)",
                req.uid,
                record["end"],
                record["attempts"],
                record["final_rows"],
                record["generated_total"],
            )
        return msgs

    def _process_one_msg(self, msg: BaseTTSBackendMsg) -> None:
        if isinstance(msg, BatchTTSBackendMsg):
            for m in msg.data:
                self._process_one_msg(m)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, TTSUserMsg):
            logger.debug_rank0("Received TTS user msg: %s", msg.uid)
            input_len = len(msg.input_ids)
            max_seq_len = self.engine.max_seq_len
            if input_len >= max_seq_len:
                return logger.warning_rank0(
                    f"TTS input len {input_len} exceeds {max_seq_len}, request {msg.uid} is dropped."
                )
            self._request_stop_id(msg.sampling_params)
            self.waiting_reqs.append(msg)
        else:
            logger.error(f"Unknown TTS message type: {type(msg)}")
            raise NotImplementedError

    def _request_stop_id(self, sampling_params) -> int:
        """End token a request stops on: EOA for utterances, span-end for edit spans."""
        end_token = sampling_params.end_token
        if end_token == "utterance":
            return self.eoa_id
        if end_token == "span":
            if self.span_end_id is None:
                raise ValueError("end_token='span' needs a checkpoint with a span-end token (span_end_id)")
            return self.span_end_id
        raise ValueError(f"end_token must be 'utterance' or 'span', got {end_token!r}")

    def _speaker_background_token(self, clean: bool) -> int:
        # Imported lazily: zonos3.tts pulls in TTSLLM, which imports this module.
        from zonos3.tts.prompt import speaker_background_token_id

        return speaker_background_token_id(
            self.text_vocab,
            self.speaking_rate_num_buckets,
            self.quality_bucket_counts,
            clean,
            self.speaker_background_num_buckets,
            self.accurate_mode_num_buckets,
        )

    def _accurate_mode_token(self) -> int:
        from zonos3.tts.prompt import accurate_mode_token_id

        return accurate_mode_token_id(
            self.text_vocab,
            self.speaking_rate_num_buckets,
            self.quality_bucket_counts,
            self.speaker_background_num_buckets,
            self.accurate_mode_num_buckets,
        )

    def _with_speaker_frames(
        self,
        input_ids: torch.Tensor,
        speaker_token_position: int,
    ) -> torch.Tensor:
        """Ensure the canonical speaker slot and marker frames lead the prompt.

        Train-time prompt layout: speaker slot at frame 0, then the speaker-background marker, then (optionally) the
        accurate-mode marker. Checkpoints with these markers get the former request defaults: the noisy-background
        marker (clean_speaker_background=False) and accurate mode on.
        """
        needs_background = self.speaker_background_num_buckets > 0
        needs_accurate = self.accurate_mode_num_buckets > 0

        def _marker(text_token: int) -> torch.Tensor:
            return _make_marker_slot(
                input_ids,
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                text_token=text_token,
            )

        has_slot = speaker_token_position == 0 and _has_reserved_speaker_slot(
            input_ids,
            n_codebooks=self.n_codebooks,
            audio_pad_id=self.audio_pad_id,
            text_vocab=self.text_vocab,
        )
        if not has_slot:
            prefix = [
                _make_speaker_slot(
                    input_ids,
                    n_codebooks=self.n_codebooks,
                    audio_pad_id=self.audio_pad_id,
                    text_vocab=self.text_vocab,
                )
            ]
            if needs_background:
                prefix.append(_marker(self._speaker_background_token(False)))
                if needs_accurate:
                    prefix.append(_marker(self._accurate_mode_token()))
            return torch.cat(prefix + [input_ids], dim=0)

        if not needs_background:
            return input_ids

        background_ids = {
            self._speaker_background_token(True),
            self._speaker_background_token(False),
        }
        frame1 = _frame_text_token(input_ids, 1, n_codebooks=self.n_codebooks, audio_pad_id=self.audio_pad_id)
        if frame1 not in background_ids:
            inserted = [_marker(self._speaker_background_token(False))]
            if needs_accurate:
                inserted.append(_marker(self._accurate_mode_token()))
            return torch.cat([input_ids[:1]] + inserted + [input_ids[1:]], dim=0)

        if needs_accurate:
            frame2 = _frame_text_token(input_ids, 2, n_codebooks=self.n_codebooks, audio_pad_id=self.audio_pad_id)
            if frame2 != self._accurate_mode_token():
                return torch.cat([input_ids[:2], _marker(self._accurate_mode_token()), input_ids[2:]], dim=0)

        return input_ids

    def _schedule_prefill(self) -> TTSBatch | None:
        """Schedule a prefill batch from waiting requests."""
        if not self.waiting_reqs or self.table_manager.available_size == 0:
            return None

        reqs = []
        total_tokens = 0

        while self.waiting_reqs and self.table_manager.available_size > 0:
            if total_tokens >= self.prefill_budget:
                break

            msg = self.waiting_reqs[0]
            input_ids = msg.input_ids
            speaker_token_position = msg.speaker_token_position
            if msg.speaker_embedding is not None:
                input_ids = self._with_speaker_frames(input_ids, speaker_token_position)
                speaker_token_position = 0
            # Condition rows keep their offset in the tokenizer's prompt; rows prepended here (a speaker slot or
            # marker the tokenizer did not add) shift them.
            condition_values = (
                [int(v) for v in msg.condition_values] if getattr(msg, "condition_values", None) is not None else None
            )
            condition_token_start = int(getattr(msg, "condition_token_start", -1))
            if condition_values is not None:
                if self.condition_count != len(condition_values):
                    raise ValueError(
                        f"request {msg.uid}: {len(condition_values)} condition rows for a model with {self.condition_count}"
                    )
                condition_token_start += len(input_ids) - len(msg.input_ids)
            elif self.condition_count:
                raise ValueError(f"request {msg.uid}: the tokenizer sent no condition rows for a condition-row model")

            # Controls CFG: a guided request runs a paired unconditional twin (same
            # prompt + speaker, but no hidden steering delta) that decodes in lockstep, so
            # it consumes a second table slot / cache handle.
            controls_cfg = (
                (msg.steering_delta is not None or getattr(msg, "steer_release_delta", None) is not None)
                and msg.speaker_embedding is not None
                and float(msg.sampling_params.controls_cfg_scale) != 1.0
            )
            speaker_cfg = float(msg.sampling_params.speaker_cfg_scale) != 1.0
            if speaker_cfg and controls_cfg:
                raise ValueError("Speaker and controls CFG cannot be combined")
            # Speaker CFG on either path: the twin is the prompt without its speaker conditioning (the reference
            # prefix or the reserved speaker row), no speaker embedding, all-unknown condition rows = the evaluation
            # generator's twin (zonos3.speaker_cfg.speaker_cfg_twin).
            cfg_twin_spec = (
                speaker_cfg_twin(
                    msg,
                    input_ids,
                    n_codebooks=self.n_codebooks,
                    audio_pad_id=self.audio_pad_id,
                    text_vocab=self.text_vocab,
                    speaker_token_position=speaker_token_position,
                    condition_values=condition_values,
                    condition_token_start=condition_token_start,
                    condition_unknowns=self.condition_unknowns,
                )
                if speaker_cfg
                else None
            )
            twin_input_ids = cfg_twin_spec.input_ids if speaker_cfg else input_ids
            use_cfg = speaker_cfg or controls_cfg
            needed_slots = 2 if use_cfg else 1
            if self.table_manager.available_size < needed_slots:
                break  # don't start a CFG pair unless both slots are free

            input_len = len(input_ids)
            prompt_len = input_len
            max_seq_len = self.engine.max_seq_len
            if prompt_len >= max_seq_len:
                logger.warning_rank0(f"TTS input len {prompt_len} exceeds {max_seq_len}, request {msg.uid} is dropped.")
                self.waiting_reqs.pop(0)
                continue

            accent_slot, accent_scale = 0, 0.0
            if msg.accent is not None:
                try:
                    if self.engine.accent_adapters is None:
                        raise ValueError("this server loads no accent adapters")
                    accent_slot = self.engine.accent_adapters.slot(msg.accent)
                except ValueError as exc:
                    logger.warning_rank0(f"TTS request {msg.uid} is dropped: {exc}")
                    self.waiting_reqs.pop(0)
                    continue
                accent_scale = float(msg.accent_strength)

            max_output_len = max_seq_len - prompt_len
            sampling_params = msg.sampling_params
            if msg.sampling_params.max_tokens > max_output_len:
                sampling_params = replace(msg.sampling_params, max_tokens=max_output_len)
                logger.debug_rank0(
                    "Adjust TTS max_tokens to %d for request %d.",
                    max_output_len,
                    msg.uid,
                )

            pair_len = input_len + (len(twin_input_ids) if use_cfg else 0)
            if total_tokens + pair_len > self.prefill_budget and reqs:
                break  # Don't exceed budget unless this is the first request

            self.waiting_reqs.pop(0)
            total_tokens += pair_len

            # Create TTSReq
            table_idx = self.table_manager.allocate()
            cache_handle = self.cache_manager.allocate_new_handle()

            # Copy input_ids to token pool
            input_ids_i32 = input_ids.to(torch.int32)
            self.token_pool[table_idx, :input_len].copy_(input_ids_i32.pin_memory().to(self.device), non_blocking=True)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TTS %d prefill: input_len=%d, input_ids shape=%s",
                    msg.uid,
                    input_len,
                    input_ids_i32.shape,
                )
                logger.debug("TTS %d   first frame: %s", msg.uid, input_ids_i32[0].tolist())
                logger.debug("TTS %d   last frame: %s", msg.uid, input_ids_i32[-1].tolist())
                if input_len > 20:
                    logger.debug("TTS %d   frame at -20: %s", msg.uid, input_ids_i32[-20].tolist())
                    logger.debug("TTS %d   frame at -17: %s", msg.uid, input_ids_i32[-17].tolist())

            stop_id = self._request_stop_id(msg.sampling_params)
            # Alignment probe: the text span is located on the final prompt (after any speaker rows the scheduler added).
            align_probe = getattr(msg, "align_probe", None)
            history_rows = int(getattr(msg, "history_rows", 0))
            span_follower = None
            if getattr(msg, "laci", None) is not None:
                if align_probe is not None:
                    raise ValueError(f"request {msg.uid}: an align_probe and LACI are exclusive")
                align_probe = self._laci_probe(msg)
                span_follower = self.laci_requests[msg.uid].session.follower  # fed by the session's observer
            elif getattr(msg, "steer_follow", None) is not None and align_probe is None:
                from zonos3_sgl.tts.align_probe import AlignProbe

                span_follower = self._span_follower(msg, origin=history_rows)
                if span_follower is not None:
                    align_probe = AlignProbe(self.laci_heads[msg.steer_follow].heads, span_follower)
            if align_probe is not None:
                from zonos3.tts.probe import locate_text_span

                align_probe.text_start, align_probe.text_len = locate_text_span(input_ids, self.n_codebooks)
            rng = None
            if msg.sampling_params.seed is not None:
                rng = torch.Generator(device=self.device)
                rng.manual_seed(msg.sampling_params.seed)

            req = TTSReq(
                input_ids=input_ids_i32,
                table_idx=table_idx,
                cached_len=0,
                output_len=sampling_params.max_tokens,
                uid=msg.uid,
                sampling_params=sampling_params,
                cache_handle=cache_handle,
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                stop_id=stop_id,
                eoa_policy=self.eoa_policy,
                rng=rng,
                speaker_embedding=msg.speaker_embedding,
                speaker_token_position=speaker_token_position,
                steering_delta=msg.steering_delta,
                steer_schedule=getattr(msg, "steer_schedule", None),
                steer_release_delta=getattr(msg, "steer_release_delta", None),
                steer_release=getattr(msg, "steer_release", None),
                span_follower=span_follower,
                condition_values=condition_values,
                condition_token_start=condition_token_start,
                decode_context_codes=getattr(msg, "decode_context_codes", None),
                prefill_rows=getattr(msg, "prefill_rows", None),
                align_probe=align_probe,
                history_rows=history_rows,
                cfg_scale=(
                    float(msg.sampling_params.speaker_cfg_scale)
                    if speaker_cfg
                    else float(msg.sampling_params.controls_cfg_scale)
                    if controls_cfg
                    else 1.0
                ),
                accent_slot=accent_slot,
                accent_scale=accent_scale,
            )
            reqs.append(req)

            if use_cfg:
                # Speaker CFG uses the target prompt with the full reference removed.
                # Each branch has its own prompt length, positions and KV cache.
                # The shared sampled target frames advance both branches together.
                # Controls CFG retains its unsteered twin prompt. Under speaker CFG
                # the hidden-space steering (steering delta, span schedule / follower,
                # accent adapter) applies to both branches: it is part of the generation,
                # not of the contrast, so guidance amplifies only the speaker contrast.
                twin_table_idx = self.table_manager.allocate()
                twin_cache_handle = self.cache_manager.allocate_new_handle()
                twin_ids_i32 = twin_input_ids.to(torch.int32)
                self.token_pool[twin_table_idx, : len(twin_ids_i32)].copy_(
                    twin_ids_i32.pin_memory().to(self.device), non_blocking=True
                )
                twin = TTSReq(
                    input_ids=twin_ids_i32,
                    table_idx=twin_table_idx,
                    cached_len=0,
                    output_len=sampling_params.max_tokens,
                    uid=msg.uid,
                    sampling_params=sampling_params,
                    cache_handle=twin_cache_handle,
                    n_codebooks=self.n_codebooks,
                    audio_pad_id=self.audio_pad_id,
                    stop_id=stop_id,
                    eoa_policy=self.eoa_policy,
                    rng=None,
                    # The speaker-CFG twin removes speaker conditioning and shifts its all-unknown condition
                    # rows back by the removed rows; the controls-CFG twin keeps its speaker and condition rows.
                    # Hidden-space steering is shared under speaker CFG so guidance amplifies only the speaker
                    # contrast (the follower's decisions are per frame, shared with the twin).
                    speaker_embedding=None if speaker_cfg else msg.speaker_embedding,
                    speaker_token_position=-1 if speaker_cfg else speaker_token_position,
                    steering_delta=msg.steering_delta if speaker_cfg else None,
                    steer_schedule=getattr(msg, "steer_schedule", None) if speaker_cfg else None,
                    steer_release_delta=getattr(msg, "steer_release_delta", None) if speaker_cfg else None,
                    steer_release=getattr(msg, "steer_release", None) if speaker_cfg else None,
                    span_follower=span_follower if speaker_cfg else None,
                    condition_values=cfg_twin_spec.condition_values if speaker_cfg else condition_values,
                    condition_token_start=cfg_twin_spec.condition_token_start if speaker_cfg else condition_token_start,
                    is_cfg_uncond=True,
                    accent_slot=accent_slot,
                    accent_scale=accent_scale,
                )
                req.cfg_twin = twin
                reqs.append(twin)

        if not reqs:
            return None

        return TTSBatch(reqs=reqs, phase="prefill")

    def _prepare_batch(self, batch: TTSBatch) -> TTSForwardInput:
        """Prepare a batch for forward pass."""
        needed_size = sum(r.extend_len for r in batch.reqs)
        batch.out_loc = self.cache_manager.allocate(needed_size)

        # Pad batch if needed for CUDA graph
        if padding_size := self.engine.graph_runner.pad_batch(batch):
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)

        # For TTS we load full frames
        load_ranges = [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        write_ranges = [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]

        load_indices = _make_3d_indices(self.token_pool, load_ranges)
        write_indices = _make_3d_indices(self.token_pool, write_ranges)

        # Write out_loc to page table
        self.page_table.view(-1)[load_indices // self.frame_width] = batch.out_loc
        self.engine.attn_backend.prepare_metadata(batch)

        return TTSForwardInput(
            batch=batch,
            sample_args=self.tts_sampler.prepare(
                batch, token_pool=self.token_pool, stop_bias=self._laci_stop_bias(batch)
            ),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> TTSForwardInput | None:
        """Schedule next batch (prefill first, then decode)."""
        # Scheduling writes the token pool (a new request's prompt rows, possibly in a slot just freed) and reads it
        # (repetition history) on this stream, while the step still running on the engine stream writes its sampled
        # rows: order this stream's work after that step on the GPU. The host does not wait (sampling never syncs),
        # so it prepares the next step while the GPU runs the current one.
        torch.cuda.current_stream().wait_stream(self.engine.stream)
        batch = self._schedule_prefill() or self.decode_manager.schedule_next_batch()
        return self._prepare_batch(batch) if batch else None

    def _log_batch_stats(self, batch: TTSBatch) -> None:
        self._stats_batches += 1
        self._stats_tokens += batch.size

        if batch.is_prefill:
            logger.info_rank0(
                "TTS Prefill: reqs=%d, extend_frames=%d, waiting=%d, decode_running=%d",
                batch.size,
                sum(r.extend_len for r in batch.reqs),
                len(self.waiting_reqs),
                len(self.decode_manager.running_reqs),
            )

        now = time.perf_counter()
        elapsed = now - self._stats_start
        if elapsed >= 5.0:
            logger.info_rank0(
                "TTS Throughput (%.1fs window): batches=%d, decode_steps=%d, %.1f frames/s, "
                "decode_running=%d, waiting=%d",
                elapsed,
                self._stats_batches,
                self._stats_tokens,
                self._stats_tokens / elapsed,
                len(self.decode_manager.running_reqs),
                len(self.waiting_reqs),
            )
            self._stats_start = now
            self._stats_batches = 0
            self._stats_tokens = 0

    def _load_token_ids(self, input: TTSForwardInput) -> None:
        """Load token IDs from pool to batch."""
        # Load full frames for TTS
        flat_pool = self.token_pool.view(-1, self.frame_width)
        frame_indices = input.load_indices // self.frame_width
        input.batch.input_ids = flat_pool[frame_indices]

        # Debug: verify loaded tokens
        if logger.isEnabledFor(logging.DEBUG) and input.batch.is_decode and input.batch.size > 0:
            req = input.batch.reqs[0]
            if req.total_generated < 10:
                loaded_tokens = input.batch.input_ids[0].tolist()
                logger.debug(
                    "Load frame %d: loaded from frame_index %d, tokens = %s",
                    req.total_generated,
                    frame_indices[0].item(),
                    loaded_tokens,
                )

    def _write_token_ids(self, input: TTSForwardInput, output: TTSForwardOutput) -> None:
        """Write generated tokens back to pool."""
        flat_pool = self.token_pool.view(-1, self.frame_width)
        frame_indices = input.write_indices // self.frame_width
        flat_pool[frame_indices] = output.next_tokens_gpu

        # Debug: verify what was written
        if logger.isEnabledFor(logging.DEBUG) and input.batch.size > 0:
            req = input.batch.reqs[0]
            if req.total_generated < 10:
                written_tokens = flat_pool[frame_indices[0]].tolist()
                logger.debug(
                    "Write frame %d: wrote tokens %s at frame_index %d",
                    req.total_generated,
                    written_tokens,
                    frame_indices[0].item(),
                )

    def _cfg_pairs(self, batch: TTSBatch) -> list[tuple[int, int, float]]:
        """Return (cond_row, uncond_row, cfg_scale) for guided pairs in the batch.

        Rows index into the forward logits / sampled tokens (ordered to match
        ``batch.reqs``).
        """
        row = {id(req): i for i, req in enumerate(batch.reqs)}
        pairs: list[tuple[int, int, float]] = []
        for req in batch.reqs:
            twin = req.cfg_twin
            if twin is None or req.cfg_scale == 1.0:
                continue
            j = row.get(id(twin))
            if j is None:
                raise RuntimeError("CFG pair split across batches")
            if req.total_generated != twin.total_generated or req.remain_len != twin.remain_len:
                raise RuntimeError("CFG pair generated history is out of step")
            pairs.append((row[id(req)], j, float(req.cfg_scale)))
        return pairs

    def _apply_cfg(self, logits: torch.Tensor, batch: TTSBatch) -> torch.Tensor:
        """Combine cond/uncond logit rows: guided = uncond + scale*(cond-uncond)."""
        pairs = self._cfg_pairs(batch)
        if not pairs:
            return logits
        guided = logits.float().clone()
        for i, j, scale in pairs:
            guided[i] = guided_logits(logits[i], logits[j], scale)
        # Keep FP32 through repetition penalties, temperature and filtering.
        return guided

    def _sync_cfg_twin_tokens(self, next_tokens: torch.Tensor, batch: TTSBatch) -> None:
        """Copy each conditional req's sampled frame onto its uncond twin so both
        KV histories stay identical (lockstep); the twin's own frame is discarded."""
        for i, j, _ in self._cfg_pairs(batch):
            next_tokens[j] = next_tokens[i]

    def _sample_cfg(self, logits, sample_args, batch):
        """Sample only externally visible rows, then mirror one sample to each twin."""
        pairs = self._cfg_pairs(batch)
        if not pairs:
            return self.tts_sampler.sample(logits, sample_args)
        rows = [i for i, req in enumerate(batch.reqs) if not req.is_cfg_uncond]
        indices = make_device_tensor(rows, torch.long, logits.device)

        def select(value):
            if value is None:
                return None
            return value[rows] if value.device.type == "cpu" else value.index_select(0, indices)

        visible_args = replace(
            sample_args,
            temperatures=select(sample_args.temperatures),
            top_ks=select(sample_args.top_ks),
            top_ps=select(sample_args.top_ps),
            min_ps=select(sample_args.min_ps),
            repetition_token_ids=select(sample_args.repetition_token_ids),
            repetition_penalties=select(sample_args.repetition_penalties),
            stop_token_ids=select(sample_args.stop_token_ids),
            stop_blocked=select(sample_args.stop_blocked),
            stop_bias=select(sample_args.stop_bias),
            generators=[sample_args.generators[i] for i in rows],
        )
        sampled = self.tts_sampler.sample(logits.index_select(0, indices), visible_args)
        result = torch.empty((len(batch.reqs), sampled.shape[1]), dtype=sampled.dtype, device=sampled.device)
        result.index_copy_(0, indices, sampled)
        self._sync_cfg_twin_tokens(result, batch)
        return result

    def _prepare_speaker_conditioning(self, batch: TTSBatch) -> None:
        """Prepare per-batch speaker embeddings and injection positions."""
        if not self.speaker_enabled:
            batch.speaker_emb_values = None
            batch.speaker_token_positions = None
            return

        emb_values: list[torch.Tensor] = []
        token_positions: list[int] = []
        offset = 0

        for req in batch.reqs:
            if req.speaker_embedding is None:
                offset += req.extend_len
                continue

            rel_pos = req.speaker_token_position - req.cached_len
            if 0 <= rel_pos < req.extend_len:
                emb = req.speaker_embedding
                if emb.numel() != self.speaker_embedding_dim:
                    logger.warning_rank0(
                        "Skipping speaker embedding for uid=%d: expected dim=%d, got=%d",
                        req.uid,
                        self.speaker_embedding_dim,
                        emb.numel(),
                    )
                else:
                    emb_values.append(emb.to(torch.float32))
                    token_positions.append(offset + rel_pos)
            offset += req.extend_len

        if emb_values:
            batch.speaker_emb_values = to_device(torch.stack(emb_values, dim=0), self.device)
            batch.speaker_token_positions = make_device_tensor(token_positions, torch.long, self.device)
        else:
            batch.speaker_emb_values = None
            batch.speaker_token_positions = None

    def _prepare_condition_rows(self, batch: TTSBatch) -> None:
        """Bucket ids + packed positions of the condition rows (loudness / fmax / speaking rate) that fall inside
        this extend span, one (ids, positions) pair per table (the model gathers each table once)."""
        if not self.condition_count:
            batch.condition_rows = None
            return
        values: list[list[int]] = [[] for _ in range(self.condition_count)]
        positions: list[list[int]] = [[] for _ in range(self.condition_count)]
        offset = 0
        for req in batch.reqs:
            if req.condition_values is not None and req.condition_token_start >= 0:
                for index in range(self.condition_count):
                    rel = req.condition_token_start + index - req.cached_len
                    if 0 <= rel < req.extend_len:
                        values[index].append(int(req.condition_values[index]))
                        positions[index].append(offset + rel)
            offset += req.extend_len
        rows = []
        for index in range(self.condition_count):
            if positions[index]:
                rows.append(
                    (
                        make_device_tensor(values[index], torch.long, self.device),
                        make_device_tensor(positions[index], torch.long, self.device),
                    )
                )
            else:
                rows.append(None)
        batch.condition_rows = rows if any(row is not None for row in rows) else None

    def _prepare_accents(self, batch: TTSBatch) -> None:
        """Per-token accent slots and strengths, aligned with batch.input_ids."""
        if self.engine.accent_adapters is None:
            batch.accent_slots = batch.accent_scales = None
            return
        reqs = batch.padded_reqs
        lengths = torch.tensor([req.extend_len for req in reqs])
        slots = torch.tensor([req.accent_slot for req in reqs], dtype=torch.int32)
        scales = torch.tensor([req.accent_scale for req in reqs], dtype=torch.float32)
        batch.accent_slots = to_device(torch.repeat_interleave(slots, lengths), self.device)
        batch.accent_scales = to_device(torch.repeat_interleave(scales, lengths), self.device)

    def _forward(self, forward_input: TTSForwardInput) -> TTSForwardOutput:
        """Run forward pass on batch."""
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        self._prepare_speaker_conditioning(batch)
        self._prepare_condition_rows(batch)
        self._prepare_accents(batch)

        # Debug: show input shape and first request's input
        if logger.isEnabledFor(logging.DEBUG):
            if batch.is_prefill and batch.size > 0:
                logger.debug("Forward prefill input_ids shape: %s", batch.input_ids.shape)
            elif batch.is_decode and batch.size > 0:
                req = batch.reqs[0]
                if req.total_generated < 10:
                    input_tokens = batch.input_ids[0].tolist()
                    eoa_id = self.eoa_id
                    eoa_positions = [i for i, t in enumerate(input_tokens[: self.n_codebooks]) if t == eoa_id]
                    logger.debug(
                        "Forward decode step %d: input tokens = %s",
                        req.total_generated,
                        input_tokens,
                    )
                    if eoa_positions:
                        logger.debug(
                            "Forward WARNING: EOA token in INPUT at codebook positions %s!",
                            eoa_positions,
                        )
                    positions = batch.attn_metadata.positions
                    logger.debug(
                        "Forward positions: %s",
                        (positions[: batch.size].tolist() if hasattr(positions, "tolist") else positions),
                    )

        # Forward through engine - get multi-codebook logits
        logits = self.engine.forward_batch_tts(batch)

        if self.sample_fp32_logits:
            logits = logits.float()
        # Classifier-free guidance (speaker / controls / tag): combine cond/uncond rows.
        logits = self._apply_cfg(logits, batch)

        # Debug: check logits stats for first 5 frames
        if logger.isEnabledFor(logging.DEBUG) and batch.size > 0:
            req = batch.reqs[0]
            if req.total_generated < 5:
                logger.debug(
                    "Forward frame %d: logits shape=%s, min=%.2f, max=%.2f, mean=%.2f",
                    req.total_generated,
                    logits.shape,
                    logits.min().item(),
                    logits.max().item(),
                    logits.mean().item(),
                )

        # Sample tokens
        next_tokens = self._sample_cfg(logits, sample_args, batch)

        _apply_cold_start_delay_mask(
            next_tokens,
            batch,
            n_codebooks=self.n_codebooks,
            audio_pad_id=self.audio_pad_id,
        )

        # Mirror the conditional token onto its uncond twin so both stay in step.
        self._sync_cfg_twin_tokens(next_tokens, batch)

        # Copy to CPU
        next_tokens_cpu = torch.empty_like(next_tokens, device="cpu", pin_memory=True)
        copy_done = torch.cuda.Event()
        next_tokens_cpu.copy_(next_tokens, non_blocking=True)
        # Alignment-probe readout (argmax, centroid per row) rides on the same event: no extra host sync.
        probe_cpu = None
        if batch.probe is not None:
            probe_cpu = torch.empty((batch.size, 2), dtype=torch.float32, device="cpu", pin_memory=True)
            probe_cpu.copy_(batch.probe.out[: batch.size], non_blocking=True)
        copy_done.record()

        self._write_token_ids(
            forward_input,
            TTSForwardOutput(next_tokens, next_tokens_cpu, copy_done),
        )

        # Mark all requests as available for decode
        for req in batch.reqs:
            req.complete_one()
        self.decode_manager.add_reqs(forward_input.batch.reqs)

        return TTSForwardOutput(next_tokens, next_tokens_cpu, copy_done, probe_cpu)

    def run_when_idle(self) -> None:
        """Called when scheduler is idle."""
        logger.info_rank0("TTS Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: TTSForwardData | None) -> TTSForwardData | None:
        """Main loop with overlapping scheduling and execution."""
        blocking = not (last_data or self.waiting_reqs or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            self._log_batch_stats(forward_input.batch)
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        """Non-overlapping loop."""
        blocking = not (self.waiting_reqs or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            self._log_batch_stats(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """Run scheduler loop forever."""
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()
