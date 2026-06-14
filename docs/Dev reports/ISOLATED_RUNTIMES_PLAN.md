# Isolated Runtimes Plan

## Goal

Keep ComfyUI in one modern main runtime while routing fragile engines into
their own pinned Python environments.

## Why

- `transformers` major-version breaks now hit engines differently
- one broken HF stack should not poison the whole node runtime
- per-engine rollback is simpler than global dependency rollback

## User impact

- isolated runtimes duplicate dependencies on disk
- large packages such as PyTorch will usually be installed again inside the worker runtime
- this must be warned in:
  - docs
  - node tooltips
  - runtime creation logs

## Contract

ComfyUI process:
- owns UI, node orchestration, model discovery, caching policy
- prepares normalized request payloads
- launches a worker only for engines configured as `runtime_mode=isolated`

Worker process:
- runs in a dedicated Python env
- receives a JSON request
- performs load / generate / unload actions
- returns JSON response + output artifact paths

## Initial profiles

- `vibevoice_transformers4`
- `qwen3_tts_transformers5`
- `step_audio_editx_transformers5`
- `moss_tts_transformers5`

## First target

VibeVoice / KugelAudio

Reason:
- highest dependency fragility
- strongest evidence that global `transformers 5` compatibility is not enough
- likely template for future HF-heavy engines

## Implementation order

1. Add config + registry support for runtime routing
2. Add launcher + request/response protocol
3. Add VibeVoice worker entrypoint
4. Add opt-in routing from unified interface
5. Add user-facing runtime path configuration
6. Expand to Qwen / MOSS / Step Audio only after VibeVoice works

## Non-goals

- no silent env creation
- no arbitrary command execution
- no shell-constructed user input
- no automatic migration of all engines
- no pretending disk duplication does not exist
