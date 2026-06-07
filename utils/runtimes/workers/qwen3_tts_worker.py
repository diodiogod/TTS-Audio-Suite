from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtimes.protocol import RuntimeJobResponse


def _emit(protocol_out, response: RuntimeJobResponse) -> None:
    protocol_out.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    protocol_out.flush()


def _load_ref_audio(payload: Optional[Dict[str, Any]]) -> Any:
    if payload is None:
        return None

    kind = payload.get("kind")
    if kind == "audio_path":
        return payload.get("audio_path")

    if kind == "tensor_path":
        tensor_path = payload.get("tensor_path")
        if not tensor_path:
            raise RuntimeError("tensor_path payload missing file path")
        tensor_payload = torch.load(tensor_path, map_location="cpu")
        waveform = tensor_payload["waveform"]
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()
        return waveform, int(tensor_payload.get("sample_rate", 24000))

    raise RuntimeError(f"Unsupported isolated Qwen3-TTS ref_audio kind: {kind}")


def _save_generation_result(output_path: str, wavs, sample_rate: int) -> None:
    payload_wavs = []
    wav_list = wavs if isinstance(wavs, list) else [wavs]
    for wav in wav_list:
        if isinstance(wav, torch.Tensor):
            payload_wavs.append(wav.detach().cpu().float())
        else:
            payload_wavs.append(torch.tensor(wav, dtype=torch.float32))
    torch.save({"wavs": payload_wavs, "sample_rate": int(sample_rate)}, output_path)


def _build_streamer(text: str, max_new_tokens: int):
    from engines.qwen3_tts.progress_callback import Qwen3TTSProgressStreamer

    return Qwen3TTSProgressStreamer(max_new_tokens=max_new_tokens, progress_bar=None, text_input=text)


def _ensure_windows_compiler_env() -> None:
    if os.name != "nt":
        return

    if os.environ.get("CC") and os.environ.get("CXX"):
        return

    try:
        from utils.runtimes.launcher import IsolatedRuntimeLauncher

        msvc_env = IsolatedRuntimeLauncher._get_windows_msvc_env()
    except Exception:
        msvc_env = None

    if msvc_env:
        for key in ("PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR", "VSINSTALLDIR"):
            value = msvc_env.get(key)
            if value:
                os.environ[key] = value

    cl_path = shutil.which("cl", path=os.environ.get("PATH", ""))
    if cl_path:
        os.environ.setdefault("CC", cl_path)
        os.environ.setdefault("CXX", cl_path)


@lru_cache(maxsize=8)
def _can_use_torch_compile(mode: str = "default") -> bool:
    if not hasattr(torch, "compile"):
        return False

    if not torch.cuda.is_available():
        return True

    # Use a real cached compile smoke test instead of guessing from PATH/CC.
    # The Windows environment may compile successfully even when naïve `cl`
    # detection fails, and the opposite can also happen with stale PATH entries.
    try:
        def _compile_probe(x):
            return (x + 1).relu()

        compiled = torch.compile(_compile_probe, mode=mode)
        x = torch.randn(16, 16, device="cuda")
        y = compiled(x)
        torch.cuda.synchronize()
        del compiled
        del x
        del y
        return True
    except Exception as exc:
        print(f"⚠️ Qwen3-TTS isolated runtime: torch.compile smoke test failed for mode={mode}: {exc}")
        return False


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    _ensure_windows_compiler_env()

    from engines.qwen3_tts.qwen3_tts import Qwen3TTSEngine

    engine = None

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue

        request = None
        try:
            request = json.loads(stripped)
            action = request.get("action")
            payload = request.get("payload") or {}
            request_id = request.get("request_id")

            if action == "shutdown":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"shutdown": True}, request_id=request_id))
                break

            if action == "ping":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"pong": True}, request_id=request_id))
                continue

            if action == "initialize":
                engine = Qwen3TTSEngine(
                    model_name=request.get("model_name") or "Qwen3-TTS-12Hz-1.7B-Base",
                    device=request.get("device") or "auto",
                    dtype=payload.get("dtype", "auto"),
                    attn_implementation=payload.get("attn_implementation", "auto"),
                    model_dir=payload.get("model_path"),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"model_name": request.get("model_name")},
                        request_id=request_id,
                    ),
                )
                continue

            if engine is None:
                raise RuntimeError("Qwen3-TTS worker received action before initialization")

            if action == "enable_streaming_optimizations":
                use_compile = payload.get("use_compile", True)
                compile_mode = payload.get("compile_mode", "reduce-overhead")
                if use_compile and not _can_use_torch_compile(compile_mode):
                    print(f"⚠️ Qwen3-TTS isolated runtime: torch.compile disabled because the runtime smoke test failed for mode={compile_mode}")
                    use_compile = False
                engine.enable_streaming_optimizations(
                    use_compile=use_compile,
                    use_cuda_graphs=payload.get("use_cuda_graphs", True),
                    compile_mode=compile_mode,
                    decode_window_frames=payload.get("decode_window_frames", 80),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"enabled": True, "use_compile": use_compile},
                        request_id=request_id,
                    ),
                )
                continue

            if action == "generate_custom_voice":
                max_new_tokens = payload.get("max_new_tokens", 2048)
                wavs, sr = engine.generate_custom_voice(
                    text=payload["text"],
                    language=payload.get("language", "Auto"),
                    speaker=payload.get("speaker", "Vivian"),
                    instruct=payload.get("instruct"),
                    top_k=payload.get("top_k", 50),
                    top_p=payload.get("top_p", 1.0),
                    temperature=payload.get("temperature", 0.9),
                    repetition_penalty=payload.get("repetition_penalty", 1.05),
                    max_new_tokens=max_new_tokens,
                    streamer=_build_streamer(payload["text"], max_new_tokens),
                )
                _save_generation_result(payload["output_path"], wavs, sr)
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"output_path": payload["output_path"]}, request_id=request_id))
                continue

            if action == "generate_voice_design":
                max_new_tokens = payload.get("max_new_tokens", 2048)
                wavs, sr = engine.generate_voice_design(
                    text=payload["text"],
                    language=payload.get("language", "Auto"),
                    instruct=payload["instruct"],
                    top_k=payload.get("top_k", 50),
                    top_p=payload.get("top_p", 1.0),
                    temperature=payload.get("temperature", 0.9),
                    repetition_penalty=payload.get("repetition_penalty", 1.05),
                    max_new_tokens=max_new_tokens,
                    streamer=_build_streamer(payload["text"], max_new_tokens),
                )
                _save_generation_result(payload["output_path"], wavs, sr)
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"output_path": payload["output_path"]}, request_id=request_id))
                continue

            if action == "generate_voice_clone":
                ref_audio = _load_ref_audio(payload.get("ref_audio"))
                max_new_tokens = payload.get("max_new_tokens", 2048)
                wavs, sr = engine.generate_voice_clone(
                    text=payload["text"],
                    language=payload.get("language", "Auto"),
                    ref_audio=ref_audio,
                    ref_text=payload.get("ref_text"),
                    x_vector_only_mode=payload.get("x_vector_only_mode", False),
                    top_k=payload.get("top_k", 50),
                    top_p=payload.get("top_p", 1.0),
                    temperature=payload.get("temperature", 0.9),
                    repetition_penalty=payload.get("repetition_penalty", 1.05),
                    max_new_tokens=max_new_tokens,
                    streamer=_build_streamer(payload["text"], max_new_tokens),
                )
                _save_generation_result(payload["output_path"], wavs, sr)
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"output_path": payload["output_path"]}, request_id=request_id))
                continue

            raise RuntimeError(f"Unsupported Qwen3-TTS worker action '{action}'")

        except Exception as exc:
            _emit(
                protocol_out,
                RuntimeJobResponse(
                    ok=False,
                    error=f"{exc}\n{traceback.format_exc()}",
                    request_id=(request.get("request_id") if isinstance(request, dict) else None),
                ),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
