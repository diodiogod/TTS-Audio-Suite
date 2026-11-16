"""Frozen dual codec: native SSL teacher -> semantic Q1, acoustic residual RVQ.

Each independent lane retains both external-rate teacher audio and full-file
resampled codec audio. Packing external hop samples into the second channel is
lossless and only aligns the two sample grids for the existing batch scheduler.
The teacher sees 20-second lanes, including the same 10-second real left context
as the acoustic encoder. It retains native bidirectional attention within a lane.
"""

import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from zonos3.codec.data_processing.independent_batcher import IndependentBatcher
from zonos3.codec.data_processing.torch_utils import compile_fixed
from zonos3.codec.hybrid10_dual_encoding.inference import extract_ssl
from zonos3.codec.hybrid10_dual_encoding.model import MossCodec
from zonos3.codec.resample_exact import resample_exact
from zonos3.codec.s1dac.losses import SemanticTeacher


class MossAutoencoder:
    def __init__(
        self,
        args,
        *,
        generator_path: str,
        teacher_dir: str,
        step: int,
        codebook_sizes: list[int],
        name: str,
        checkpoint_sha256: str,
        device: str,
    ):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("Production encoding requires an allocated GPU")
        torch.cuda.set_device(self.device)
        self.receipt = {
            "step": step,
            "codebook_sizes": list(codebook_sizes),
            "output_name": name,
            "sha256": checkpoint_sha256,
            "checkpoint": generator_path,
        }
        self.teacher_dir = teacher_dir
        state = torch.load(generator_path, map_location="cpu", weights_only=True, mmap=True)
        cfg = state["config"]["model"]
        self.cfg = cfg
        assert state["step"] == self.receipt["step"]
        self.codebook_sizes = self.receipt["codebook_sizes"]
        assert self.codebook_sizes == [cfg["semantic_codebook_size"]] + [cfg["residual_codebook_size"]] * 9
        assert cfg["residual_codebooks"] == 9 and cfg["frame_rate"] == 15 and cfg["dual_encoding"]
        assert (
            cfg["sample_rate"],
            cfg["internal_sample_rate"],
            cfg["internal_hop_length"],
            cfg["external_samples_per_frame"],
        ) == (44100, 44160, 2944, 2940)
        model = MossCodec(cfg)
        model.load_state_dict(state["generator"], strict=True, assign=True)
        model.eval().requires_grad_(False)
        self.resample = model.to_internal.cpu()
        # Both semantic modules are required for token encoding.
        del model.decoder, model.to_external, model.to_internal
        self.model = model.to(self.device)
        self.teacher = (
            SemanticTeacher(teacher_dir, layer=state["config"]["teacher"]["layer"], dtype=torch.bfloat16)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        del state
        self.sample_rate = 44100
        self.feature_dim = 10
        self.name = self.receipt["output_name"]
        self.batch_size = args.hybrid_batch_size
        self.chunk_seconds = args.hybrid_chunk_seconds
        self.context_seconds = args.hybrid_context_seconds
        self.execution = (
            "cuda_graph_acoustic_encoder_native_ssl_and_dual_rvq"
            if args.hybrid_cuda_graph
            else "eager_native_ssl_and_dual_rvq"
        )
        self.packer = IndependentBatcher(
            torch=torch,
            device=self.device,
            sample_rate=44160,
            hop_length=2944,
            feature_dim=10,
            prepare=self._prepare,
            run=self._run,
            batch_size=self.batch_size,
            chunk_seconds=self.chunk_seconds,
            continuation_context_seconds=self.context_seconds,
            window=32,
            output_dtype=np.int16,
            label=self.name,
            input_dtype=torch.float32,
            input_channels=2,
        )
        self.optimized = None
        print(json.dumps(dict(event="model_loaded", **self.metadata())), flush=True)
        with torch.inference_mode():
            g = torch.Generator(device=self.device).manual_seed(728)
            warmup = (
                torch.randn((self.batch_size, 1, self.packer.chunk_samples), generator=g, device=self.device) * 0.05
            )
            expected = self._encoder(warmup).clone()
            self.optimized = compile_fixed(
                torch,
                self._encoder,
                warmup,
                enabled=args.hybrid_cuda_graph,
                cache_name=self.name,
                strategy="cuda_graph",
            )
            assert torch.equal(self.optimized(warmup) if self.optimized else self._encoder(warmup), expected), (
                "Graph altered acoustic latent"
            )
            for scale in [0.05, 0.1]:
                packed = (
                    torch.randn((self.batch_size, 2, self.packer.chunk_samples), generator=g, device=self.device)
                    * scale
                )
                assert torch.equal(self._run(packed), self._codes(packed)), "Graph/native dual tokens disagree"
        report = {
            "event": "inference_validation_passed",
            "step": self.receipt["step"],
            "codebooks": 10,
            "codebook_sizes": self.codebook_sizes,
            "graph_token_agreement": 1.0,
            "native_quantize_dual": True,
            "teacher_layer": 16,
            "batch_size": self.batch_size,
            "chunk_seconds": self.chunk_seconds,
            "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        print(json.dumps(report), flush=True)

    def _external(self, packed):
        b = packed.shape[0]
        return packed[:, 1:2].reshape(b, 1, -1, 2944)[..., :2940].reshape(b, 1, -1)

    def _encoder(self, audio):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.encoder(audio)

    def _dual(self, packed, optimized):
        # Uses the exact CPU feature preprocessor and frozen teacher from training.
        audio = packed[:, 0:1].contiguous()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ssl = extract_ssl(self._external(packed), self.teacher)
            latent = self.optimized(audio) if optimized and self.optimized is not None else self.model.encoder(audio)
            result = self.model.quantize_dual(latent, ssl)
        return result["codes"].transpose(1, 2)

    def _codes(self, packed):
        return self._dual(packed, False)

    def _run(self, packed):
        return self._dual(packed, True)

    def _prepare(self, audio, sample_rate):
        if sample_rate != 44100:
            raise ValueError("Expected 44100 Hz")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[0] != 1 or not audio.shape[-1]:
            raise ValueError("Expected nonempty mono audio")
        if not np.isfinite(audio).all():
            raise ValueError("Nonfinite source audio")
        with torch.inference_mode():
            wave = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            frames = (wave.shape[-1] + 2939) // 2940
            internal = resample_exact(wave, self.resample)
            internal = F.pad(internal, (0, frames * 2944 - internal.shape[-1]))
            external = F.pad(wave, (0, frames * 2940 - wave.shape[-1])).reshape(1, 1, frames, 2940)
            external = F.pad(external, (0, 4)).reshape(1, 1, frames * 2944)
            return torch.cat([internal, external], dim=1)

    def metadata(self):
        return {
            "name": self.name,
            "model": "s1_s2_dac/hybrid_dual_encoding",
            "checkpoint_step": self.receipt["step"],
            "checkpoint_sha256": self.receipt["sha256"],
            "checkpoint": self.receipt["checkpoint"],
            "sample_rate": 44100,
            "internal_sample_rate": 44160,
            "hop_length": 2940,
            "frame_rate": 15,
            "n_codebooks": 10,
            "codebook_sizes": self.codebook_sizes,
            "semantic_codebook_index": 0,
            "representation": "discrete_codes",
            "layout": ["time", "codebook"],
            "output_dtype": np.dtype(np.int16).str,
            "batch_size": self.batch_size,
            "chunk_seconds": self.chunk_seconds,
            "continuation_context_seconds": self.context_seconds,
            "resample_policy": "full_file_before_hop_aligned_chunking",
            "long_file_policy": "overlap_left_context_then_discard_context_tokens",
            "precision": "fp32_weights_bf16_encoder_teacher_semantic_modules_fp32_rvq",
            "execution": self.execution,
            "ssl_teacher_layer": 16,
            "ssl_teacher": self.teacher_dir,
            "ssl_preprocessing": "native_training_44100_to_16000_fbank_stride2",
            "ssl_context_policy": "native_bidirectional_teacher_on_independent_padded_lanes",
        }

    def reference_metadata(self, audio, sample_rate, output):
        expected = (audio.shape[-1] + 2939) // 2940
        if output.shape != (expected, 10):
            raise ValueError(f"Incorrect token shape {output.shape}, expected {(expected, 10)}")
        if output.min() < 0 or np.any(output >= np.asarray(self.codebook_sizes)[None, :]):
            raise ValueError("Invalid codebook index")
        return {
            "audio_samples": int(audio.shape[-1]),
            "audio_sample_rate": sample_rate,
            "checkpoint_step": self.receipt["step"],
        }

    def encode_stream(self, samples):
        count = 0
        seconds = 0.0
        start = time.monotonic()
        for sample, codes in self.packer.encode_stream(samples):
            if sample.error is None:
                self.reference_metadata(sample.audio, sample.sample_rate, codes)
                seconds += sample.audio.shape[-1] / sample.sample_rate
            count += 1
            if count % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "encoding_progress",
                            "files": count,
                            "audio_hours": seconds / 3600,
                            "audio_seconds_per_wall_second": seconds / (time.monotonic() - start),
                        }
                    ),
                    flush=True,
                )
            yield sample, codes

    def encode(self, audio, sample_rate):
        return next(iter(self.encode_stream([SimpleNamespace(audio=audio, sample_rate=sample_rate, error=None)])))[1]
