"""
ZONOS3-Mini fast engine.

Runs a native GenerateRequest through the vendored `zonos3_sgl` engine (Zyphra's CUDA-graph scheduler) instead of the
plain-PyTorch reference loop. The request is planned by the same shared rules the reference uses (`zonos3.tts.plan`),
so the prompt rows, conditioning, steering, CFG twin, LACI and short-text handling are the ones the reference path
builds; only the LM execution differs. Audio is decoded and post-processed by the reference's own functions.

Needs CUDA and Triton; `availability()` reports why it cannot run.
"""

import importlib.util
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# The qualified-mode arithmetic of the release checkpoint (servers/sgl/scripts/env.sh upstream), with the Triton
# kernel for the MoE experts. Read by zonos3_sgl at construction.
_ENGINE_ENV = {
    "ZONOS3_USE_TRAINING_RMSNORM": "1",
    "ZONOS3_USE_TRAINING_ROUTER_WEIGHTS": "1",
    "ZONOS3_USE_TRAINING_SONIC_EXPERTS": "0",
    "ZONOS3_USE_TRAINING_ATTNRES": "1",
    "ZONOS3_USE_TRAINING_ROPE": "1",
}

# LACI reads this many text keys per request inside the decode graph; longer texts decode without the graph.
PROBE_MAX_TEXT = 1024


def availability(device: str, dtype: torch.dtype) -> Optional[str]:
    """None when the fast engine can run on ``device`` in ``dtype``; otherwise the reason it cannot."""
    if not str(device).startswith("cuda"):
        return "it needs a CUDA device"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    index = torch.device(device).index
    if index not in (None, 0):
        return "it runs on cuda:0 only"
    if dtype != torch.bfloat16:
        return "it runs in bfloat16 only"
    if importlib.util.find_spec("triton") is None:
        return "the triton package (triton-windows on Windows) is not installed"
    # cudaMallocAsync's captured allocations fail Triton's pointer checks and can
    # abort Python while the failed graph is destroyed. Check before constructing it.
    if torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync":
        return (
            "its Triton CUDA graphs require PyTorch's native GPU allocator; "
            "cudaMallocAsync can crash graph capture. Restart ComfyUI with "
            "--disable-cuda-malloc to use Fast, or select Reference"
        )
    return None


class _Interrupted(Exception):
    pass


class Zonos3MiniFastRunner:
    """One `zonos3_sgl` scheduler bound to a model folder, a CUDA device and a dtype."""

    def __init__(self, root: str, device: str, dtype: torch.dtype, assets, steer_bundle: str, laci_heads: Dict[str, str]):
        for key, value in _ENGINE_ENV.items():
            os.environ[key] = value
        from zonos3.tts import plan as P
        from zonos3_sgl.distributed import DistributedInfo
        from zonos3_sgl.scheduler import SchedulerConfig

        self.P = P
        self.assets = assets
        lm_dir = os.path.join(root, "lm", "zonos3_mini")
        with open(os.path.join(lm_dir, "params.json"), encoding="utf-8") as handle:
            params = json.load(handle)
        # Rows per request (reference prefix + text + generated frames): the checkpoint's whole context, as the
        # reference implementation plans with. A request and its CFG twin each get that many KV pages; at 32768 rows
        # this costs about 0.3 GB more than a 4096-row pool (measured) and no speed.
        context_rows = int(params.get("model", params)["max_seqlen"])
        self.device = torch.device("cuda:0")  # zonos3_sgl binds its single rank to cuda:0
        config = _config_class(SchedulerConfig)(
            model_path=lm_dir,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            attention_backend="fa",
            cache_type="naive",
            max_running_req=2,  # a request and its CFG twin
            cuda_graph_max_bs=2,
            max_seq_len_override=context_rows,
            num_page_override=2 * context_rows,
            phoneme_rows=os.path.join(lm_dir, "text_rows", "pause_tokens_s5000.pt"),
            laci_heads=dict(laci_heads),
            align_probe_max_text=PROBE_MAX_TEXT,
            align_probe_max_masked=1,
            align_probe_max_keys=context_rows,
            steer_bundle=steer_bundle,
        )
        self._previous_stream = torch.cuda.current_stream(self.device)
        # Built in inference mode, like every later call: graph capture updates the CUDA generator's state in place,
        # and that state is an inference tensor once anything (the reference-clip encoder) seeded it in that mode.
        with torch.inference_mode():
            self.scheduler = _OfflineScheduler(config)
        self.context = _FastContext(self.scheduler, config, params, assets, self.device)

    # ------------------------------------------------------------------ generation

    @torch.inference_mode()
    def generate(self, request: dict, should_stop, on_frame) -> Tuple[np.ndarray, Optional[dict]]:
        """float32 mono PCM at 44.1 kHz and the LACI record (None without LACI). ``should_stop()`` is polled once per
        scheduler step; ``on_frame()`` is called once per generated row."""
        from zonos3.reference.synth import decode_rows, fade_out, output_gain, output_pcm

        P, ctx, scheduler = self.P, self.context, self.scheduler
        plan = P.plan_prompt(ctx, request, self._speaker)
        if plan.accent is not None:
            raise ValueError("accent adapters are not available")
        req = plan.request
        rows, eos_frame, laci = scheduler.run(self._message(plan), should_stop, on_frame)
        pcm = decode_rows(
            ctx.lm_view,
            self.assets,
            rows,
            prefill_rows=plan.prefill_rows,
            eos_frame=eos_frame,
            context_codes=plan.context_codes,
            device=self.device,
            decoder=req.decoder,
        )
        gain = output_gain(ctx.config.audio_codec, req.headroom_db, req.decoder)
        return fade_out(output_pcm(pcm, gain), float(req.fade_out_ms)).astype(np.float32), laci

    def _speaker(self, req, mode):
        """The request's speaker inputs: an embedding file or inline reference codes (the adapter sends no clips)."""
        from zonos3.reference.synth import decode_base64_blob, load_embedding_vector

        P, ctx = self.P, self.context
        if req.speaker_embedding_id or req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b:
            raise ValueError("speaker ids need a server")
        if req.speaker_audio_base64:
            raise ValueError("speaker_audio_base64 is not supported; send an embedding or reference codes")
        embedding = None
        if mode == "reference":
            if req.speaker_embedding_base64:
                raise ValueError(
                    "This checkpoint conditions on raw reference audio and does not accept "
                    "speaker embedding files or embedding blends."
                )
        elif req.speaker_embedding_base64:
            embedding = load_embedding_vector(
                decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                expected_dim=ctx.config.speaker_embedding_dim,
                file_name=req.speaker_embedding_name,
            )
        codes = None if req.speaker_reference_codes is None else P.reference_codes(ctx, req.speaker_reference_codes)
        return P.Speaker(mode, embedding, codes)

    def _message(self, plan):
        """The scheduler message of a planned request: what the server's tokenizer worker sends for it."""
        from zonos3_sgl.message import TTSUserMsg
        from zonos3_sgl.message.tts import TTSSamplingParams

        req, cond = plan.request, plan.cond
        follow = plan.follow_tokens is not None
        return TTSUserMsg(
            uid=0,
            input_ids=cond.input_ids,
            speaker_reference_prefix_length=int(plan.reference_prefix_length),
            sampling_params=TTSSamplingParams(
                temperature=float(req.temperature),
                topk=int(req.topk),
                top_p=float(req.top_p),
                min_p=float(req.min_p),
                max_tokens=int(plan.max_tokens),
                repetition_window=int(req.repetition_window),
                repetition_penalty=float(req.repetition_penalty),
                repetition_codebooks=int(req.repetition_codebooks),
                seed=int(plan.seed),
                controls_cfg_scale=float(req.controls_cfg_scale),
                speaker_cfg_scale=float(req.speaker_cfg_scale),
                stream_decode=None,
                decoder=req.decoder,
            ),
            speaker_embedding=cond.speaker_embedding,
            speaker_token_position=max(cond.speaker_row, 0) if cond.speaker_embedding is not None else -1,
            steering_delta=plan.hidden_delta,
            steer_schedule=plan.schedule,
            steer_release_delta=plan.release_delta,
            steer_release=None if plan.release is None else list(plan.release),
            steer_follow=plan.language if follow else None,
            steer_follow_tokens=plan.follow_tokens if follow else None,
            steer_follow_gate=plan.follow_gate if follow else None,
            condition_values=cond.condition_values,
            condition_token_start=int(cond.condition_token_start),
            prefill_rows=None if plan.prefill_rows is None else plan.prefill_rows.to(torch.int32).contiguous(),
            laci=plan.laci,
            laci_short_text=plan.short_text,
        )

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Free the engine's GPU memory (weights, KV pool, GDN state, CUDA graphs) so it can be built again."""
        import gc

        import zonos3_sgl.core as core
        import zonos3_sgl.distributed.info as info
        import zonos3_sgl.layers.rotary as rotary

        try:
            self.scheduler.shutdown()
        finally:
            # zonos3_sgl keeps one engine per process in module globals; clear them with the engine
            core._GLOBAL_CTX = None
            info._TP_INFO = None
            rotary.get_rope.cache_clear()  # RoPE tables on the GPU
            rotary._ROPE_DEVICE = None
            torch.cuda.set_stream(self._previous_stream)
            self.scheduler = self.context = None
            gc.collect()
            torch.cuda.empty_cache()


def _config_class(base):
    """SchedulerConfig with the steering bundle path the engine reads (upstream adds it on its server config)."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class FastConfig(base):
        steer_bundle: Optional[str] = None

    return FastConfig


def _scheduler_base():
    from zonos3_sgl.scheduler.scheduler import TTSScheduler

    return TTSScheduler


class _OfflineSchedulerMixin:
    """One request at a time through the scheduler loop, with a stop check and a per-row callback."""

    def run(self, message, should_stop, on_frame) -> Tuple[List[torch.Tensor], Optional[int], Optional[dict]]:
        from zonos3_sgl.env import ENV

        self._pending = [message]
        self._rows: List[torch.Tensor] = []
        self._eos_frame = self._laci = None
        self._done = False
        self._should_stop, self._on_frame = should_stop, on_frame
        try:
            if ENV.DISABLE_OVERLAP_SCHEDULING:
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    while not self._done:
                        self.normal_loop()
            else:
                data = None
                while not self._done:
                    data = self.overlap_loop(data)
        except _Interrupted:
            self._abort()
            raise InterruptedError("ZONOS3-Mini generation interrupted by user") from None
        except BaseException:
            self._abort()
            raise
        return self._rows, self._eos_frame, self._laci

    def offline_receive_msg(self, blocking: bool = False):
        if self._should_stop():
            raise _Interrupted()
        pending, self._pending = self._pending, []
        if blocking and not pending:
            self._done = True
        return pending

    def offline_send_result(self, reply) -> None:
        width = self.n_codebooks + 1
        for msg in reply.data:
            if msg.finished:
                self._eos_frame, self._laci = msg.eos_frame, msg.laci
            if msg.audio_codes is not None and (not msg.finished or len(msg.audio_codes)):
                row = torch.tensor([int(v) for v in msg.audio_codes], dtype=torch.int32)
                if len(row) == width - 1:
                    row = torch.cat((row, row.new_zeros(1)))
                self._rows.append(row)
                self._on_frame()

    def _abort(self) -> None:
        """Drop every request so the next run starts from an idle scheduler."""
        torch.cuda.synchronize(self.device)
        requests = set(self.decode_manager.running_reqs) | set(self.finished_reqs)
        self.decode_manager.running_reqs.clear()
        self.finished_reqs.clear()
        self.waiting_reqs.clear()
        self.laci_requests.clear()
        for req in requests:
            self.cache_manager.free_slots(self.page_table[req.table_idx, : req.cached_len])
            self.cache_manager.free_handle(req.cache_handle)
            self.table_manager.free(req.table_idx)


def _make_offline_scheduler():
    base = _scheduler_base()

    class OfflineScheduler(_OfflineSchedulerMixin, base):
        pass

    return OfflineScheduler


def _OfflineScheduler(config):
    return _make_offline_scheduler()(config)


class _LMView:
    """The few `Zonos3LM` attributes the reference's decode helpers read."""

    def __init__(self, config):
        self.config = config


class _FastContext:
    """`zonos3.tts.plan.PromptContext` over the fast engine: the reference's context fields from the engine's model
    config, so both engines plan a request identically."""

    steer_hint = "the steering bundle failed to load"

    def __init__(self, scheduler, config, params, assets, device):
        from zonos3.generation import SAMPLE_RATE
        from zonos3.reference.lm import Zonos3LMConfig
        from zonos3.tokenizer.matched_codec import entry
        from zonos3.tts import plan as P
        from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig

        c = Zonos3LMConfig.from_params(params)
        self.config, self.assets, self.device = c, assets, device
        self.lm_view = _LMView(c)
        self.n_codebooks, self.codebook_size = c.n_codebooks, c.codebook_size
        self.audio_pad_id, self.text_vocab, self.span_end_id = c.audio_pad_id, c.text_vocab, c.span_end_id
        self.hidden_size = c.dim
        self.max_seqlen = self.max_frames = int(scheduler.engine.max_seq_len)
        self.condition_names, self.condition_bucket_counts = c.condition_names, c.condition_bucket_counts
        self.condition_edges, self.condition_flags = c.condition_edges, {}
        self.speaker_modes = (["reference"] if c.speaker_reference_prefix_enabled else []) + ["embedding"]
        self._builder = TTSPromptBuilder(
            TTSPromptConfig(
                n_codebooks=c.n_codebooks,
                codebook_sizes=c.codebook_sizes,
                audio_pad_id=c.audio_pad_id,
                text_vocab=c.text_vocab,
                text_tokenizer_spec=config.text_tokenizer_spec,
            )
        )
        self.text_tokenizer_spec = self._builder.config.text_tokenizer_spec
        self.accent_names = []
        self.steer_bundle, self.steering_sources = assets.steer_bundle, assets.steering_sources()
        self.laci_languages = tuple(assets.laci_heads)
        codec = entry(c.audio_codec)
        self.reference_max_frames = P.reference_frame_limit(c.speaker_reference_max_seconds, codec["hop"], SAMPLE_RATE)
        self.frame_rate = SAMPLE_RATE / int(codec["hop"])
        self.silence_available = codec.get("silence_frame") is not None or codec.get("family") == "dual"

    def prompt_builder(self):
        return self._builder

    def normalize_text(self, text: str, language: str, enabled: bool) -> str:
        from zonos3.tokenizer.textnorm import normalize_request_text

        return normalize_request_text(text, language, self.assets.normalizer() if enabled else None)

    def silence_frame(self) -> torch.Tensor:
        from zonos3.reference.synth import silence_frame

        return silence_frame(self.config.audio_codec, self.device)
