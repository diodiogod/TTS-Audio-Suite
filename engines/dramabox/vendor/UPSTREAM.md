# Bundled DramaBox inference and training source

This directory contains the inference-critical source copied unchanged from:

- Repository: `https://github.com/resemble-ai/DramaBox`
- Commit: `a70a5818e103c1c9fef22409c1e0c707ebf4f8a7`
- License: LTX-2 Community License Agreement in `LICENSE`

The bundled-code changes are marked inline:

- `ltx2/ltx_pipelines/utils/blocks.py`: local-only Gemma loading prevents
  Transformers from silently downloading outside TTS Audio Suite's organized
  ComfyUI model directory; staged modes can defer and release the warm prompt
  encoder between generation stages.
- `src/inference_server.py`: ComfyUI cancellation exceptions are allowed to
  propagate from progress callbacks instead of being swallowed; the official
  negative-prompt, FP8-cast, compile, and staged-memory controls are exposed to
  the suite wrapper; suite-managed PEFT LoRA loading is added for trained
  DramaBox audio adapters.
- `src/validate.py`: validation accepts the suite's separately organized
  DramaBox transformer and audio-components checkpoints.
- `src/preprocess.py`: suite-distributed pre-quantized Gemma checkpoints use
  the same bitsandbytes-aware prompt-encoder loader as DramaBox inference.
- `src/train.py`: the batch collator lives at module scope so Windows
  spawn-based DataLoader workers can serialize it; lightweight per-step
  telemetry keeps the suite's training dashboard current between normal logs.
- `ltx2/ltx_core/text_encoders/gemma/encoders/encoder_configurator.py`:
  supports both wrapped and direct SigLIP vision-tower layouts for the suite's
  newer Transformers runtime.

The official training entry points are also bundled at this pin:

- `src/preprocess.py`
- `src/train.py`
- `src/validate.py`
- `configs/training_args.example.yaml`
- `configs/val_config.example.yaml`

The suite invokes these scripts through the unified training backend. Apart
from the documented compatibility patches, training behavior stays upstream;
dataset normalization, job lifecycle, and UI wiring remain suite-side.
