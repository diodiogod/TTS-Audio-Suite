from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root_str = str(PROJECT_ROOT)
sys.path = [
    entry for entry in sys.path
    if str(Path(entry).resolve()) != project_root_str
]
sys.path.insert(0, project_root_str)

# These warnings originate in inherited third-party packages. Keep the filters
# exact and local to this worker so unrelated warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"jieba\._compat",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
    module=r"rotary_embedding_torch\..*",
)

from utils.runtimes.protocol import RuntimeJobResponse


class _WorkerProgressBar:
    """Enable Step's console token-rate reporting inside the isolated worker."""

    def update(self, delta=1):
        del delta


def _emit(stream, response):
    stream.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    stream.flush()


def main():
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    engine = None

    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            action = request.get("action")
            payload = request.get("payload") or {}
            request_id = request.get("request_id")

            if action == "shutdown":
                _emit(protocol_out, RuntimeJobResponse(ok=True, request_id=request_id))
                break
            if action == "ping":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"pong": True}, request_id=request_id))
                continue
            if action == "initialize":
                from engines.step_audio_editx.step_audio_editx_impl.model_loader import ModelSource
                from engines.step_audio_editx.step_audio_editx_impl.tokenizer import StepAudioTokenizer
                from engines.step_audio_editx.step_audio_editx_impl.tts import StepAudioTTS

                dtype = getattr(torch, payload.get("torch_dtype", "bfloat16"), torch.bfloat16)
                tokenizer = StepAudioTokenizer(
                    encoder_path=payload["model_path"],
                    model_source=ModelSource.LOCAL,
                )
                engine = StepAudioTTS(
                    model_path=payload["model_path"],
                    audio_tokenizer=tokenizer,
                    model_source=ModelSource.LOCAL,
                    quantization_config=payload.get("quantization"),
                    torch_dtype=dtype,
                    device_map=request.get("device") or "cuda",
                )
                _emit(protocol_out, RuntimeJobResponse(ok=True, request_id=request_id))
                continue
            if engine is None:
                raise RuntimeError("Step Audio EditX worker received generation before initialization")

            if action == "clone":
                audio, sample_rate = engine.clone(
                    prompt_wav_path=payload["prompt_wav_path"],
                    prompt_text=payload["prompt_text"],
                    target_text=payload["target_text"],
                    temperature=payload.get("temperature", 0.7),
                    do_sample=payload.get("do_sample", True),
                    max_new_tokens=payload.get("max_new_tokens", 1024),
                    progress_bar=_WorkerProgressBar(),
                )
            elif action == "edit":
                audio, sample_rate = engine.edit(
                    input_audio_path=payload["input_audio_path"],
                    audio_text=payload["audio_text"],
                    edit_type=payload["edit_type"],
                    edit_info=payload.get("edit_info"),
                    text=payload.get("text"),
                    temperature=payload.get("temperature", 0.7),
                    do_sample=payload.get("do_sample", True),
                    max_new_tokens=payload.get("max_new_tokens", 1024),
                    progress_bar=_WorkerProgressBar(),
                )
            else:
                raise RuntimeError(f"Unsupported Step Audio EditX worker action '{action}'")

            output_path = Path(payload["output_path"])
            torch.save({"audio": audio.detach().cpu(), "sample_rate": sample_rate}, output_path)
            _emit(protocol_out, RuntimeJobResponse(
                ok=True,
                result={"output_path": str(output_path)},
                request_id=request_id,
            ))
        except Exception as exc:
            _emit(protocol_out, RuntimeJobResponse(
                ok=False,
                error=f"{exc}\n{traceback.format_exc()}",
                request_id=request.get("request_id") if isinstance(request, dict) else None,
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
