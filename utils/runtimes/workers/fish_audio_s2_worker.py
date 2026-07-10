from __future__ import annotations

import json
import logging
import re
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtimes.protocol import RuntimeJobResponse


class _SuppressFishModelTypeWarning(logging.Filter):
    """Hide the known Fish/Transformers config-registration warning only."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return (
            "model of type `fish_qwen3_omni` to instantiate a model of type" not in message
        )


@contextmanager
def _suppress_known_fish_transformers_noise():
    transformers_filter = _SuppressFishModelTypeWarning()
    transformer_loggers = [
        logging.getLogger("transformers"),
        logging.getLogger("transformers.configuration_utils"),
        logging.getLogger("transformers.modeling_utils"),
    ]
    for logger in transformer_loggers:
        logger.addFilter(transformers_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*model of type `fish_qwen3_omni` to instantiate a model of type.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*torch\.nn\.utils\.weight_norm is deprecated.*",
                category=FutureWarning,
            )
            yield
    finally:
        for logger in transformer_loggers:
            try:
                logger.removeFilter(transformers_filter)
            except Exception:
                pass


def _patch_upstream_project_root_probe() -> None:
    """The Fish wheel omits the .project-root marker expected by its modules."""
    import pyrootutils

    pyrootutils.setup_root = lambda *args, **kwargs: PROJECT_ROOT


def _patch_fish_reference_loader_backend() -> None:
    """
    Fish still passes torchaudio.load(..., backend=...) in ReferenceLoader.
    On torchaudio 2.9+, load() routes through TorchCodec and ignores backend,
    so the old backend-selection code is stale and emits warnings.
    """
    import io
    from pathlib import Path

    import torch
    import torchaudio
    from fish_speech.inference_engine.reference_loader import ReferenceLoader

    if getattr(ReferenceLoader, "_tts_suite_backend_patch_applied", False):
        return

    def load_audio_without_legacy_backend(self, reference_audio: bytes | str, sr: int):
        if len(reference_audio) > 255 or not Path(reference_audio).exists():
            audio_data = reference_audio
            reference_audio = io.BytesIO(audio_data)

        waveform, original_sr = torchaudio.load(reference_audio)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=sr
            )
            waveform = resampler(waveform)

        return waveform.squeeze().numpy()

    ReferenceLoader.load_audio = load_audio_without_legacy_backend
    ReferenceLoader._tts_suite_backend_patch_applied = True


def _emit(stream, response):
    stream.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    stream.flush()


@contextmanager
def _suppress_noisy_fish_runtime_logs(show_compilation_time=True):
    """Keep tqdm and useful throughput lines without repeating compile timings."""
    from fish_speech.conversation import Conversation
    from loguru import logger as fish_logger

    suppressed_patterns = (
        "Override max_seq_len to ",
        "Injected Semantic IDs into Config:",
        "Loading model from ",
        "Loading sharded safetensors weights",
        "Loading single safetensors weights",
        "Model weights loaded - Status:",
        "Restored model from checkpoint",
        "Using DualARTransformer",
        "Compiling function...",
        "Split into ",
        "--- Sample ",
        "Batch text:",
        "Visualizing prompt structure:",
        "Encoded prompt shape:",
        "Audio parts shape:",
        "Audio masks non-zero count:",
        "Bandwidth achieved:",
        "VQ features:",
        "Loaded audio with ",
        "Encoded prompt:",
        "Use same references",
        "set seed:",
    )

    original_info = fish_logger.info
    original_warning = fish_logger.warning
    original_visualize = Conversation.visualize

    def _filtered_info(message, *args, **kwargs):
        rendered = str(message)
        if not show_compilation_time and "Compilation time:" in rendered:
            return
        if any(pattern in rendered for pattern in suppressed_patterns):
            return
        return original_info(message, *args, **kwargs)

    def _filtered_warning(message, *args, **kwargs):
        rendered = str(message)
        if "set seed:" in rendered:
            return
        return original_warning(message, *args, **kwargs)

    def _silent_visualize(self, *args, **kwargs):
        return None

    fish_logger.info = _filtered_info
    fish_logger.warning = _filtered_warning
    Conversation.visualize = _silent_visualize
    try:
        yield
    finally:
        fish_logger.info = original_info
        fish_logger.warning = original_warning
        Conversation.visualize = original_visualize


def _load_decoder_bf16(config_name, checkpoint_path, device, precision):
    import fish_speech.models.dac.inference as dac_inference
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_dir = Path(dac_inference.__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in key for key in state_dict):
        state_dict = {
            key.replace("generator.", ""): value
            for key, value in state_dict.items()
            if "generator." in key
        }
    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device=device, dtype=precision)
    original_encode = model.encode

    def encode_in_model_dtype(audio_data, *args, **kwargs):
        return original_encode(audio_data.to(dtype=precision), *args, **kwargs)

    model.encode = encode_in_model_dtype
    return model


def _launch_llama_queue(model_path, model_variant, quantization, device, precision, context_length, compile_model):
    import fish_speech.models.text2semantic.inference as text_inference
    from fish_speech.models.text2semantic.llama import DualARTransformer

    original_from_pretrained = DualARTransformer.from_pretrained
    original_init_model = text_inference.init_model

    def finish_model(model, decode_one_token, compile_decode):
        model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
        model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
        model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)
        model._cache_setup_done = False
        if compile_decode:
            decode_one_token = torch.compile(
                decode_one_token,
                backend="inductor",
                mode="default",
                fullgraph=True,
            )
        return model.eval(), decode_one_token

    if model_variant == "s2-pro-fp8":
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Fish S2 FP8 requires an FP8-capable CUDA GPU")
        if torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError(
                "Fish S2 FP8 requires CUDA compute capability 8.9 or newer"
            )

        def init_fp8(checkpoint_path, device, precision, compile=False):
            from fish_speech.models.text2semantic.inference import decode_one_token_ar
            from utils.runtimes.fish_audio_s2_fp8 import load_fp8_model

            model = load_fp8_model(checkpoint_path, context_length)
            model = model.to(device=device)
            return finish_model(model, decode_one_token_ar, compile)

        text_inference.init_model = init_fp8
    elif quantization in {"bnb_int8", "bnb_nf4"}:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Fish S2 BNB quantization requires an NVIDIA CUDA GPU")

        def init_bnb(checkpoint_path, device, precision, compile=False):
            from fish_speech.models.text2semantic.inference import decode_one_token_ar
            from utils.runtimes.fish_audio_s2_bnb import quantize_linear_layers

            model = original_from_pretrained(
                checkpoint_path,
                load_weights=True,
                max_length=context_length,
            )
            mode = quantization.removeprefix("bnb_")
            model, replaced = quantize_linear_layers(model, mode, device)
            print(f"Fish Audio S2: quantized {replaced} linear layers with BNB {mode.upper()}")
            model = model.to(device=device)
            return finish_model(model, decode_one_token_ar, compile)

        text_inference.init_model = init_bnb
    else:
        def load_with_context(path, load_weights=False, max_length=None, **kwargs):
            return original_from_pretrained(
                path, load_weights=load_weights, max_length=context_length, **kwargs
            )

        DualARTransformer.from_pretrained = staticmethod(load_with_context)

    try:
        return text_inference.launch_thread_safe_queue(
            checkpoint_path=model_path,
            device=device,
            precision=precision,
            compile=compile_model,
        )
    finally:
        DualARTransformer.from_pretrained = staticmethod(original_from_pretrained)
        text_inference.init_model = original_init_model


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    _patch_upstream_project_root_probe()
    engine = None
    compile_enabled = False
    first_generation = True

    for line in sys.stdin:
        if not line.strip():
            continue
        request = None
        try:
            request = json.loads(line)
            action = request.get("action")
            payload = request.get("payload") or {}
            request_id = request.get("request_id")
            if action == "shutdown":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"shutdown": True}, request_id=request_id))
                break
            if action == "initialize":
                from fish_speech.inference_engine import TTSInferenceEngine

                _patch_fish_reference_loader_backend()
                model_path = payload["model_path"]
                model_variant = payload.get("model_variant", "s2-pro")
                quantization = payload.get("quantization", "none")
                device = payload.get("device", "cuda")
                precision = torch.bfloat16 if payload.get("precision", "bfloat16") == "bfloat16" else torch.float16
                context_length = int(payload.get("context_length", 8192))
                if model_variant != "s2-pro" or quantization != "none":
                    precision = torch.bfloat16
                print("🔄 Loading Fish Audio S2 model via official runtime")
                print(f"   Path: {model_path}")
                print(f"   Device: {device} | Precision: {precision} | Context: {context_length}")
                if quantization != "none":
                    print(f"   Quantization: {quantization}")
                compile_enabled = bool(payload.get("compile", False))
                if compile_enabled:
                    print("   Compile: enabled, first load will be slower")
                    if payload.get("compile_cache_dir"):
                        print(f"   Compile cache: {payload['compile_cache_dir']}")
                with _suppress_known_fish_transformers_noise():
                    with _suppress_noisy_fish_runtime_logs():
                        text_stage_t0 = time.perf_counter()
                        print("   Stage 1/2: loading text model runtime...")
                        llama_queue = _launch_llama_queue(
                            model_path=model_path,
                            model_variant=model_variant,
                            quantization=quantization,
                            device=device,
                            precision=precision,
                            context_length=context_length,
                            compile_model=bool(payload.get("compile", False)),
                        )
                        print(f"   Stage 1/2 complete in {time.perf_counter() - text_stage_t0:.1f}s")
                        codec_stage_t0 = time.perf_counter()
                        print("   Stage 2/2: loading codec...")
                        decoder = _load_decoder_bf16(
                            config_name="modded_dac_vq",
                            checkpoint_path=str(Path(model_path) / "codec.pth"),
                            device=device,
                            precision=precision,
                        )
                        print(f"   Stage 2/2 complete in {time.perf_counter() - codec_stage_t0:.1f}s")
                engine = TTSInferenceEngine(llama_queue, decoder, precision, bool(payload.get("compile", False)))
                print("✅ Fish Audio S2 model loaded successfully")
                sample_rate = (
                    decoder.spec_transform.sample_rate
                    if hasattr(decoder, "spec_transform")
                    else decoder.sample_rate
                )
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"sample_rate": sample_rate}, request_id=request_id))
                continue
            if engine is None:
                raise RuntimeError("Fish S2 worker received generation before initialization")
            if action != "generate":
                raise RuntimeError(f"Unsupported Fish S2 action '{action}'")

            from fish_speech.utils.file import audio_to_bytes
            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

            references = []
            for reference in payload.get("references") or []:
                ref_path = reference.get("audio_path")
                ref_text = (reference.get("text") or "").strip()
                if not ref_path or not ref_text:
                    raise ValueError("Fish S2 voice cloning requires the exact reference transcript")
                references.append(ServeReferenceAudio(audio=audio_to_bytes(ref_path), text=ref_text))
            req = ServeTTSRequest(
                text=payload["text"], references=references, seed=payload.get("seed"),
                normalize=bool(payload.get("normalize", True)), streaming=False,
                chunk_length=int(payload.get("chunk_length", 200)),
                max_new_tokens=int(payload.get("max_new_tokens", 1024)),
                top_p=float(payload.get("top_p", 0.8)),
                repetition_penalty=float(payload.get("repetition_penalty", 1.1)),
                temperature=float(payload.get("temperature", 0.8)),
                use_memory_cache="on" if payload.get("cache_reference", True) else "off",
            )
            if torch.cuda.is_available() and str(payload.get("device", "cuda")).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            final = None
            if compile_enabled and first_generation:
                print("⏳ Fish Audio S2: preparing first compiled run before token generation starts")
            with _suppress_noisy_fish_runtime_logs(
                show_compilation_time=not (compile_enabled and not first_generation)
            ):
                for result in engine.inference(req):
                    if result.code == "error":
                        raise result.error or RuntimeError("Fish S2 generation failed")
                    if result.code == "final":
                        final = result.audio
            first_generation = False
            if final is None:
                raise RuntimeError("Fish S2 returned no final audio")
            sample_rate, audio = final
            peak_memory_gb = (
                torch.cuda.max_memory_reserved() / 1e9
                if torch.cuda.is_available() else 0.0
            )
            torch.save({
                "audio": torch.as_tensor(audio).float(),
                "sample_rate": int(sample_rate),
                "peak_memory_gb": float(peak_memory_gb),
            }, payload["output_path"])
            _emit(protocol_out, RuntimeJobResponse(ok=True, result={"output_path": payload["output_path"]}, request_id=request_id))
        except Exception as exc:
            _emit(protocol_out, RuntimeJobResponse(ok=False, error=f"{exc}\n{traceback.format_exc()}", request_id=request.get("request_id") if isinstance(request, dict) else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
