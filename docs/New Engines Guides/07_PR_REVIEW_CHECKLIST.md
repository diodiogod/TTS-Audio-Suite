# 07 - PR Review Checklist

Use this before merging a new engine PR.

A PR is not ready just because it generates audio.

## Required PR Evidence

The PR author or LLM should provide:

- Official model capability report.
- Existing ComfyUI reference notes, if any.
- Selected engine scope.
- Native parameters implemented.
- Native features intentionally skipped, with reasons.
- Non-native suite-added features, if any, with maintainer approval note.
- Model download path and file layout.
- Manual ComfyUI test results.
- Known limitations.

If this evidence is missing, ask for it before deep review.

## Architecture Review

Check:

- Unified nodes are thin.
- Engine-specific orchestration lives in processors/adapters.
- TTS Text processor exists when TTS Text is supported.
- SRT processor exists when SRT is supported.
- VC or ASR processors follow existing suite patterns when scoped.
- Similar existing engines were used as references, not blindly copied.

Primary reference for modern full TTS/SRT integration:

- Qwen3-TTS.

Secondary reference for wrapper/lifecycle and special post-processing:

- Step Audio EditX.

## Model Lifecycle Review

Check:

- All model loading paths use `unified_model_interface.load_model()` unless there is a documented exception.
- ComfyUI unload/reload/Clear VRAM works.
- No `__del__` destructor unloads the model after generation.
- Device movement is handled safely.
- Quantized models are not moved with unsupported `.to()` calls.
- Multiple model variants do not accumulate in VRAM unnecessarily.

## Downloads And Dependencies Review

Check:

- Official model file layout was verified.
- Downloads go to organized `ComfyUI/models/TTS/` paths.
- No hidden or automatic downloads to random cache directories.
- `install.py` or requirements changes are justified.
- Dependency pins do not break existing engines.
- License restrictions are documented.

## Feature Review

Check:

- UI exposes native model parameters only, plus clearly documented suite infrastructure options.
- Unsupported controls are not present.
- Language controls are only present when official model supports language control.
- Character tags work if text generation is supported.
- Narrator fallback works.
- Pause tags work.
- Segment parameter switching works for real parameters.
- Inline edit/post-processing behavior respects segment boundaries if used.
- Special nodes are justified by native model capability.

## SRT Review

If SRT is supported, check:

- Interrupt/cancel works in long loops.
- Timing modes use the shared timing/assembly systems correctly.
- There are no unnecessary restrictions on timing modes.
- Character switching works in SRT if character tags are supported.
- Pause tags and generated segment durations do not corrupt timing.
- Timing report/generation info is consistent with other engines.

## Cache And Progress Review

Check:

- Generated audio cache exists where applicable.
- Cache key includes all behavior-affecting parameters.
- Reference audio identity is included when relevant.
- Progress bars or progress messages exist for long generation.
- Interrupt checks are present inside long generation loops, not only before the loop.

## Audio Review

Check:

- Output tensor shape is valid for ComfyUI.
- Sample rate is correct and documented.
- Reference audio is resampled to the native rate where required.
- Audio is not accidentally squeezed from 3D to invalid 2D output.
- Pitch/speed are not changed by wrong sample-rate metadata.

## Documentation Review

Check:

- YAML-backed engine metadata is updated if engine docs changed.
- Generated docs/tables are regenerated when required.
- Model download sources are documented.
- Model layout docs are updated if needed.
- README or user docs mention important limitations.
- Tooltips are honest about license, VRAM, speed, and model limits.

## Manual Test Matrix

At minimum, test scoped features:

- Basic TTS Text generation.
- TTS Text with character tags.
- TTS Text with pause tags.
- TTS Text with parameter switching.
- SRT generation if supported.
- SRT interrupt/cancel if supported.
- Voice Changer if supported.
- ASR if supported.
- Clear VRAM / unload models, then generate again.
- Repeat identical generation to check cache behavior.
- Missing input error paths.

## Final Merge Rule

Do not merge until the implementation passes the required parity checklist or the remaining gaps are explicitly accepted by the maintainer.
