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
- SRT processor structure.
- Interrupt handling.
- Progress feedback.
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

## Implementation Order

Follow this order:

1. Confirm scope.
2. Verify model download layout.
3. Add or update downloader/model layout logic.
4. Build the engine wrapper around official inference.
5. Register the unified model factory.
6. Build the adapter.
7. Build the main processor.
8. Build SRT, VC, ASR, or special processors if scoped.
9. Add the engine configuration node UI.
10. Wire node and adapter registration.
11. Add segment parameter support if the engine has native parameters.
12. Add generated audio cache where applicable.
13. Add interrupt checks in long loops.
14. Add progress feedback for long generation.
15. Update docs/YAML metadata.
16. Run manual ComfyUI tests.
17. Run the required parity checklist.

## Architecture Rule

Unified nodes should stay thin.

Do not put hundreds of lines of engine-specific orchestration into:

- `nodes/unified/tts_text_node.py`
- `nodes/unified/tts_srt_node.py`
- `nodes/unified/voice_changer_node.py`
- `nodes/unified/asr_transcribe_node.py`

Engine-specific orchestration belongs in processors and adapters.

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
