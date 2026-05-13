# 02 - Check Existing ComfyUI Implementations

After official model research, check whether someone already made a ComfyUI implementation.

This is useful, but dangerous if followed blindly.

## What The User Should Ask

Copy this to the LLM:

```text
Search for existing ComfyUI implementations of this model.

If you find any, clone them into:

IgnoredForGitHubDocs/For_reference/[ENGINE_NAME]/

Study them as references only. Do not treat them as the source of truth. Compare every feature and parameter against the official model implementation.

Produce notes that answer:

1. Which ComfyUI implementations exist?
2. What files or patterns are useful as reference?
3. What dependencies or install problems do they reveal?
4. What UI controls do they expose?
5. Which controls are native official model features?
6. Which controls are artificial additions?
7. Which parts should not be copied into TTS Audio Suite?
```

## Where To Put References

Clone references here:

```text
IgnoredForGitHubDocs/For_reference/[ENGINE_NAME]/
```

Do not place reference repos in the main package folders. They are study material, not suite code.

## How To Use Them

Use existing ComfyUI implementations to learn:

- How the model loads in a ComfyUI environment.
- Which dependencies break easily.
- Which model files are expected.
- Which audio formats are accepted.
- Which UI patterns users may expect.
- Which edge cases others already hit.

But always compare back to the official implementation.

## What Not To Copy Blindly

Do not copy:

- Fake parameters that are not official model features.
- Random post-processing presented as native model behavior.
- Bad model download paths.
- Silent downloads into cache directories.
- Large engine logic stuffed directly into ComfyUI node UI code.
- Dependency pins that break other engines without analysis.
- UI controls that imply unsupported capabilities.

## Maintainer Discussion Required

If a ComfyUI implementation adds something useful but non-native, discuss it first in an issue or PR.

Examples:

- Artificial speed control.
- Post-processing emotion controls.
- Non-native denoise controls.
- Any special workflow that changes model behavior beyond official inference.

The default implementation should expose native model capabilities first.

---

Navigation: [Guide Hub](README.md) | Previous: [01 - Research The Official Model](01_FIRST_ASK_THE_LLM_TO_RESEARCH_THE_OFFICIAL_MODEL.md) | Next: [03 - Decide Engine Scope](03_DECIDE_ENGINE_SCOPE.md)
