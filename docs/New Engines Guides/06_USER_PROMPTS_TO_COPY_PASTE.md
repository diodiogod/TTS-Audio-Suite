# 06 - User Prompts To Copy Paste

These prompts are for a user guiding an LLM.

Use them in order. Do not skip the research prompts.

## Prompt 1 - Official Model Research

```text
Do not write code yet.

I want to add this model/engine to TTS Audio Suite:

[PASTE OFFICIAL LINK]

Read the official repo, model card, docs, examples, license, dependency files, model download instructions, and inference scripts.

Produce a capability report for TTS Audio Suite. It must explain:

- What task the model supports: TTS Text, SRT TTS, Voice Changer, ASR, audio editing, or special feature.
- Native official parameters.
- Native official features.
- Unsupported features that should not be exposed as UI controls.
- Required model files and expected folder layout.
- Sample rate and audio format.
- Dependencies and likely conflicts.
- License restrictions.
- Whether this needs a special node beyond unified nodes.

Do not implement anything until this report is complete.
```

## Prompt 2 - Existing ComfyUI Implementations

```text
Search for existing ComfyUI implementations of this model.

If you find any, clone them into:

IgnoredForGitHubDocs/For_reference/[ENGINE_NAME]/

Study them only as references. Compare every feature against the official implementation.

Produce notes explaining:

- Which implementations exist.
- Which files/patterns are useful.
- Which dependencies or install issues they reveal.
- Which UI controls are native model features.
- Which UI controls are artificial or should not be copied.
- What should be discussed with the maintainer before adding.

Do not treat third-party ComfyUI code as the source of truth.
```

## Prompt 3 - Decide Scope

```text
Using the official capability report and ComfyUI reference notes, decide the integration scope.

Do not write code yet.

Create a scope summary with:

- Whether to support Unified TTS Text.
- Whether to support Unified SRT TTS.
- Whether to support Voice Changer.
- Whether to support ASR.
- Whether a special node is needed.
- Native parameters to expose.
- Native features to implement now.
- Native features intentionally skipped, with reasons.
- Non-native suite-added features proposed, if any.
- Existing engine references to inspect.
- Manual tests required.
```

## Prompt 4 - Implementation Plan

```text
Create an implementation plan for this engine in TTS Audio Suite.

Read first:

- PROJECT_INDEX.md
- docs/New Engines Guides/NEW_ENGINE_IMPLEMENTATION_GUIDE.md
- docs/New Engines Guides/fails_to_avoid_TTS_Engine_Implementation.md

Use Qwen3-TTS as the primary modern reference for full TTS/SRT architecture and suite parity.
Use Step Audio EditX as a secondary reference for wrapper/model lifecycle and special post-processing patterns.

Do not copy references blindly. The official implementation decides capabilities.

The plan must include files/modules to change, model download approach, model lifecycle approach, processor/adapter design, tag support, cache support, SRT/VC/ASR support if scoped, docs updates, and manual tests.
```

## Prompt 5 - Implement First Phase

```text
Implement the first phase only.

Focus on:

- Model download/layout.
- Engine wrapper.
- Unified model factory registration.
- Adapter skeleton with real generation path.
- Engine node UI for native official parameters only.

Do not add fake parameters. Do not implement special non-native features unless they were explicitly approved.

After editing, summarize changed files and what still needs implementation.
```

## Prompt 6 - Review For Suite Parity

```text
Review this implementation against:

- docs/New Engines Guides/05_REQUIRED_PARITY_CHECKLIST.md
- docs/New Engines Guides/fails_to_avoid_TTS_Engine_Implementation.md

Be strict. Find missing architecture, cache, VRAM, interrupt, audio shape, sample rate, tag, SRT, docs, and fake-parameter issues.

Return findings ordered by severity with file references.
```

## Prompt 7 - Check Fake Or Non-Native Parameters

```text
Compare the engine UI and adapter parameters against the official model implementation.

List every exposed parameter and mark it as:

- Native official model parameter.
- Suite infrastructure parameter.
- Non-native enhancement.
- Unsupported/fake and should be removed.

If any non-native enhancement exists, explain whether it was approved by the maintainer.
```

## Prompt 8 - Prepare PR Summary

```text
Prepare a PR summary for this engine integration.

Include:

- Official model capability summary.
- Existing ComfyUI references studied.
- Supported node types.
- Native features implemented.
- Native features skipped, with reasons.
- Any non-native suite-added features, with approval note.
- Model download paths.
- Manual ComfyUI test results.
- Known limitations.
```
