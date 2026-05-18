# 03 - Decide Engine Scope

Before coding, decide exactly what the integration should include.

This prevents half-integrations where basic TTS works but the required SRT path, cache, tags, VRAM, or docs are missing.

## What The User Should Ask

Copy this to the LLM:

```text
Using the official capability report and ComfyUI reference notes, decide the TTS Audio Suite scope for this engine.

Do not write code yet.

Produce a short scope document with:

1. Which node family should be supported?
   - If this is TTS, include both Unified TTS Text and Unified SRT TTS.
2. Which native features will be implemented now?
3. Which native features will be intentionally skipped, and why?
4. Which non-native suite-added features are proposed, if any?
5. Which existing engine should be used as implementation reference?
6. What files/modules will likely need changes?
7. What manual tests will prove the integration works?
```

## Node Types To Choose From

Decide whether the engine needs:

- TTS integration: Unified TTS Text plus Unified SRT TTS. These are a mandatory pair for TTS engines.
- Voice Changer / voice conversion.
- ASR Transcribe.
- A special feature node.
- No integration.

Do not add a special node just because it looks interesting. Add one only when the official model has a capability that does not fit the unified nodes.

## Native Features vs Suite Enhancements

Separate these clearly.

Native model features:

- Supported by official model code or docs.
- Should be implemented first when practical.
- Can be exposed in the engine node UI.

Suite-added enhancements:

- Added by TTS Audio Suite around the model.
- May be useful, but must be explicit.
- Need maintainer discussion if they change behavior or user expectations.

## Written Scope Required

The LLM should write a scope before implementation.

A good scope looks like this:

```text
Engine scope:
- Supports TTS: yes, with both Unified TTS Text and Unified SRT TTS
- SRT approach: generate each subtitle independently and assemble timing with AudioAssemblyEngine
- Supports Voice Changer: no, official model has no voice conversion mode
- Supports ASR: no
- Needs special node: yes, official model has native voice design from text description

Native parameters to expose:
- seed
- temperature
- top_p
- language

Do not expose:
- speed, because official model has no speed parameter

Primary reference:
- Qwen3-TTS for TTS/SRT architecture

Manual tests:
- TTS Text basic generation
- TTS Text with character tags
- TTS Text with pause tags
- SRT generation with interrupt test
- Clear VRAM then regenerate
```

If the scope is vague, do not code yet.

---

Navigation: [Guide Hub](README.md) | Previous: [02 - Check Existing ComfyUI Implementations](02_CHECK_EXISTING_COMFYUI_IMPLEMENTATIONS.md) | Next: [04 - Implementation Order For LLM](04_IMPLEMENTATION_ORDER_FOR_LLM.md)
