# 04 - Implementation Order For LLM

Use this order when the capability report and scope are complete.

Do not start from random files. Follow the architecture.

## What The User Should Ask

Copy this to the LLM:

```text
Implement this engine using the TTS Audio Suite architecture.

Before editing files, read:

- PROJECT_INDEX.md
- docs/New Engines Guides/NEW_ENGINE_IMPLEMENTATION_GUIDE.md
- docs/New Engines Guides/fails_to_avoid_TTS_Engine_Implementation.md

Use Qwen3-TTS as the primary modern reference for full TTS/SRT architecture and suite parity.
Use Step Audio EditX as a secondary reference for wrapper/model lifecycle and special post-processing patterns.
If another existing engine is closer to this model, inspect that too.

Do not copy any reference blindly. The official model implementation decides native capabilities.
```

## Reference Engines

Primary reference:

- `engines/adapters/qwen3_tts_adapter.py`
- `nodes/qwen3_tts/qwen3_tts_processor.py`
- `nodes/qwen3_tts/qwen3_tts_srt_processor.py`
- `nodes/engines/qwen3_tts_engine_node.py`

Use Qwen3-TTS for:

- Adapter/processor separation.
- Character handling.
- Pause tags.
- Segment parameters.
- Cache integration.
- SRT processor structure that reuses the main TTS processor for generation.
- Interrupt handling.
- Progress feedback.
- Console observability parity, such as input-text echo and useful live generation progress.
- Unified model loading.

Secondary reference:

- `engines/step_audio_editx/`
- `engines/adapters/step_audio_editx_adapter.py`
- `nodes/step_audio_editx/`
- `nodes/engines/step_audio_editx_engine_node.py`

Use Step Audio EditX for:

- Wrapper/model lifecycle patterns.
- Special post-processing behavior.
- Engine-specific extra node patterns.
- Model loading edge cases.

Long-form reference:

- `engines/vibevoice_engine/`
- `engines/adapters/vibevoice_adapter.py`
- `nodes/vibevoice/vibevoice_processor.py`

Use VibeVoice when the target engine is long-form and needs duration-aware segmentation behavior:

- Prefer model-appropriate duration/minute-based segmentation when required.
- Do not force short-form `max_chars_per_chunk` behavior if it conflicts with native long-form generation strategy.
- If the suite already provides chunk splitting / chunk combination controls in the unified node, reuse those. Do not add engine-local chunk-silence or chunk-combination controls unless the official model has a clearly separate native long-form mechanism.

## Implementation Order

Follow this order:

1. Confirm scope.
2. Verify model download layout.
3. Add or update downloader/model layout logic.
4. Build the engine wrapper around official inference.
5. Register the unified model factory.
6. Build the adapter.
7. Build the main processor.
8. For TTS engines, build the SRT processor too. The SRT processor should call the main TTS processor/generation path for each subtitle instead of duplicating generation logic. Build VC, ASR, or special processors only if scoped.
9. Add the engine configuration node UI.
10. Wire node and adapter registration.
11. Add segment parameter support if the engine has native parameters.
12. Add generated audio cache where applicable.
13. Add interrupt checks in long loops.
14. Add progress feedback for long generation.
15. Update docs/YAML metadata.
16. Run automated and live ComfyUI validation. Use FL-MCP-assisted validation when it is installed and connected; otherwise perform the same checks manually. Follow `tests/FL_MCP_VALIDATION.md`.
17. Run the required parity checklist.

## Live ComfyUI Validation Rule

Passing imports or pytest is not enough for a new engine. Validate it in the canonical Windows ComfyUI installation after implementation.

If [ComfyUI_FL-MCP](https://github.com/filliptm/ComfyUI_FL-MCP) is available, the LLM should use it to inspect and operate the live ComfyUI instance. Treat it as an optional test driver, not a project dependency and not a substitute for the existing test suite.

The LLM should:

- After changing Python code, restart the canonical Windows ComfyUI process before testing. An already-running process still has the old modules loaded.
- Use PowerShell to identify the process listening on port `8188`, verify its command line belongs to the canonical ComfyUI `main.py`, stop only that process, and relaunch it with the canonical Windows Python.
- Wait for `http://127.0.0.1:8188/system_stats` to respond before using FL-MCP.
- Refresh the existing ComfyUI browser tab after restart and confirm the FL-MCP browser bridge has reconnected before calling canvas-only tools.
- Confirm the engine node and the relevant unified node are registered.
- Load or construct the smallest useful workflow.
- Inspect workflow JSON for the expected node types, links, and widget values.
- Queue the workflow and wait for completion.
- Inspect execution history and report the full actionable error if execution fails.
- Confirm the expected audio output artifact exists.
- Capture a canvas screenshot for UI and broken-node inspection.
- Exercise TTS Text and SRT for every TTS engine, plus any other scoped capability.
- Record what was actually tested, what was skipped, and why.

Screenshots prove only visible workflow state. They do not prove that generation succeeded or that audio is correct. Execution history and output artifacts are required evidence, and the user must still judge subjective audio quality.

Do not install FL-MCP, alter its safety settings, or enable destructive tools unless the user authorizes it. If FL-MCP is unavailable, report that fact and follow the manual fallback in `tests/FL_MCP_VALIDATION.md`.

Repeat the edit, restart, reconnect, and validation cycle after every implementation fix that changes imported Python code. Frontend-only changes may require a hard browser refresh as well. Do not claim that a fix was tested against a ComfyUI process started before the fix was written.

## Architecture Rule

Unified nodes should stay thin.

Reference engines are examples only. Every engine must have dedicated processors and adapters; share only engine-neutral utilities.

Do not put hundreds of lines of engine-specific orchestration into:

- `nodes/unified/tts_text_node.py`
- `nodes/unified/tts_srt_node.py`
- `nodes/unified/voice_changer_node.py`
- `nodes/unified/asr_transcribe_node.py`

Engine-specific orchestration belongs in processors and adapters.

For TTS engines, SRT support is mandatory. It is the suite's subtitle-driven TTS flow, not a native capability the model has to expose.

## SRT Reuses TTS Generation

Follow the Qwen3-TTS pattern: the SRT processor is the timing/orchestration layer, not a second implementation of the engine.

The SRT processor should:

- Import or instantiate the engine's main TTS processor.
- Parse SRT subtitles and loop through subtitle entries.
- Build the subtitle-specific voice mapping and seed/parameter context.
- Call the main TTS processor, usually `processor.process_text(...)`, for each subtitle's text.
- Reuse the same adapter, cache keys, character handling, pause tags, parameter switching, progress hooks, and model lifecycle used by normal TTS Text generation.
- Assemble subtitle audio with the suite timing systems, such as `AudioAssemblyEngine`, after generation.

The SRT processor should not:

- Call the raw model directly through a separate generation path.
- Reimplement character switching, pause tags, parameter switching, cache, or reference-audio handling differently from the TTS processor.
- Add separate fake SRT-only parameters that do not exist in the normal TTS path or the suite timing layer.

## Chunking Rule

If an engine needs long-text chunking, prefer the suite chunking and shared chunk combiner unless the official model exposes a real native long-form/chunking system that is materially different.

Do not:

- Add duplicate engine-node controls for silence between chunks, chunk combination method, or similar suite-owned chunk-assembly behavior.
- Bypass the shared chunk combiner with a custom engine-local join path unless there is a documented official reason.

## Registration Areas To Check

The LLM should inspect and update the relevant registration files for the chosen scope:

- `nodes.py`
- `engines/adapters/__init__.py`
- `utils/models/unified_model_interface.py`
- `utils/models/engine_registry.py`
- `utils/text/segment_parameters.py`
- `docs/Dev reports/tts_audio_suite_engines.yaml`
- `docs/Dev reports/tts_audio_suite_aux_models.yaml` if helper models are added

Only update files that are actually needed for the scoped integration.

---

Navigation: [Guide Hub](README.md) | Previous: [03 - Decide Engine Scope](03_DECIDE_ENGINE_SCOPE.md) | Next: [05 - Required Parity Checklist](05_REQUIRED_PARITY_CHECKLIST.md)
