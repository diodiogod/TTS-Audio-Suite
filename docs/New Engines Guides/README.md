# So You Want To Add A New Engine?

This folder is the start-here guide for adding a new engine to TTS Audio Suite with help from an LLM.

The goal is simple: do not let the LLM jump straight into coding. A new engine is not ready just because it can generate one audio clip. It must fit the suite architecture, model loading, cache, tags, SRT timing, docs, and ComfyUI lifecycle.

## Start Here

If you want support for a new TTS engine, Voice Changer, ASR, or special audio model, follow these guides in order.

For this suite, TTS support means both **Unified TTS Text** and **Unified SRT TTS**. SRT is our own text-to-subtitle generation flow around a TTS engine, not an optional native model feature.

1. [First Ask The LLM To Research The Official Model](01_FIRST_ASK_THE_LLM_TO_RESEARCH_THE_OFFICIAL_MODEL.md)
2. [Check Existing ComfyUI Implementations](02_CHECK_EXISTING_COMFYUI_IMPLEMENTATIONS.md)
3. [Decide Engine Scope](03_DECIDE_ENGINE_SCOPE.md)
4. [Implementation Order For LLM](04_IMPLEMENTATION_ORDER_FOR_LLM.md)
5. [Required Parity Checklist](05_REQUIRED_PARITY_CHECKLIST.md)
6. [User Prompts To Copy Paste](06_USER_PROMPTS_TO_COPY_PASTE.md)
7. [PR Review Checklist](07_PR_REVIEW_CHECKLIST.md)

Keep these technical references nearby:

- [Deep technical implementation guide](NEW_ENGINE_IMPLEMENTATION_GUIDE.md)
- [Known traps and failures](fails_to_avoid_TTS_Engine_Implementation.md)

## The Rule

The official model implementation is the source of truth.

Existing ComfyUI implementations are useful references, but they are not authorities. They may add fake or artificial features that the official model does not support. Do not copy those by default.

Example: if another implementation adds a `speed` parameter by stretching audio after generation, that is not a native model feature. It may still be useful, but it must be discussed with the maintainer first in an issue or PR.

## What To Tell Your LLM First

Copy this before asking for implementation:

```text
Do not write code yet.

I want to add a new engine to TTS Audio Suite. First, read these files:

- PROJECT_INDEX.md
- docs/New Engines Guides/README.md
- docs/New Engines Guides/01_FIRST_ASK_THE_LLM_TO_RESEARCH_THE_OFFICIAL_MODEL.md
- docs/New Engines Guides/02_CHECK_EXISTING_COMFYUI_IMPLEMENTATIONS.md
- docs/New Engines Guides/03_DECIDE_ENGINE_SCOPE.md
- docs/New Engines Guides/NEW_ENGINE_IMPLEMENTATION_GUIDE.md
- docs/New Engines Guides/fails_to_avoid_TTS_Engine_Implementation.md

Then research the official model implementation and produce a capability report. Do not start coding until the report explains the native model features, required nodes, model files, sample rate, dependencies, license, and supported parameters.
```

## Reference Engines

Use these references after the official model research is done:

- **Qwen3-TTS**: primary modern reference for full TTS/SRT architecture and suite parity.
- **Step Audio EditX**: secondary reference for wrapper/model lifecycle patterns and special post-processing cases.

Do not copy either blindly. The new engine must be scoped from the official implementation first.

## Definition Of Done

A new engine is not done until the answer to these questions is yes:

- Does it use the suite architecture instead of stuffing engine logic into unified nodes?
- Does it load models through the unified model interface?
- Does model unload/reload/Clear VRAM work?
- Does it support generated audio cache where applicable?
- Does it handle ComfyUI audio shape and sample rate correctly?
- Does it wire character tags, pause tags, and parameter switching where applicable?
- If it is a TTS engine, does its SRT path support interrupt handling and correct timing modes?
- Does the UI avoid fake unsupported controls?
- Are docs and engine metadata updated?
- Did someone test it manually in ComfyUI?

If the answer is no, the PR is not ready.
