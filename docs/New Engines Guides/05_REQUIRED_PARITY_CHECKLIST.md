# 05 - Required Parity Checklist

Paste this checklist to the LLM before review.

If the LLM cannot answer these questions, the implementation is not ready.

## Copy-Paste Checklist

```text
Review the new engine implementation against this TTS Audio Suite parity checklist.

Architecture:
- Did you use a processor instead of putting engine logic in unified nodes?
- Are unified nodes only doing thin routing/delegation?
- Is engine-specific orchestration in processors/adapters?

Model loading and lifecycle:
- Did you use unified_model_interface.load_model() in all model-loading paths?
- Does Clear VRAM / unload / reload work?
- Did you avoid __del__ destructors that auto-unload models after generation?
- Are model variants unloaded or reused correctly instead of accumulating VRAM?

Downloads and dependencies:
- Did you verify the real Hugging Face / ModelScope / official file layout?
- Do models download into organized ComfyUI/models/TTS/ folders?
- Did you prevent silent downloads into random cache folders?
- Did you document dependency conflicts or install.py changes?
- Did you explicitly decide whether this engine belongs in Main Environment or needs runtime isolation?
- If runtime isolation is needed, did you document the default mode and the reason in YAML/README?

Audio format:
- Did you verify the model native sample rate?
- Does output use valid ComfyUI audio dict/tensor shape?
- Did you resample only when needed?
- Did you avoid pitch/speed bugs from wrong sample-rate metadata?

Generated audio cache:
- Did you add generated audio cache where applicable?
- Does the cache key include text, voice/reference audio identity, model variant, seed, and all behavior-affecting params?
- Did you test generating the same input twice and confirm cache behavior?

Text features:
- Did you wire character tags if this node supports text generation?
- Did you handle narrator fallback?
- Did you wire pause tags?
- Did you wire parameter switching for real native parameters?
- Did you avoid adding fake unsupported parameters?
- Did you remove misleading controls like language selection if unsupported?
- Did you reuse the suite chunking/chunk-combination controls instead of inventing duplicate engine-local chunk UI or chunk-silence parameters?

SRT features:
- If this is a TTS engine, did you implement the SRT path?
- Did you add interrupt checks in long SRT loops?
- Does SRT use the shared timing/assembly systems correctly?
- Did you avoid unnecessary timing-mode restrictions?
- Does character switching work in SRT if text character tags are supported?

UX:
- Did you add progress feedback for long generation?
- Did you preserve useful console observability parity instead of making the engine run silently (for example input-text echo and live it/s/progress)?
- Are error messages clear when required inputs are missing?
- Are tooltips honest about limitations, license, VRAM, and expected text length?

Docs and metadata:
- Did you update YAML-backed engine metadata?
- Did you regenerate docs/tables if metadata changed?
- Did you document model paths and download sources?
- Did you add examples or notes for special native features?
- If the engine needs a secondary environment, does the engine table clearly show that?

Manual tests:
- Did TTS Text work?
- Did SRT TTS work for every TTS engine?
- Did Voice Changer work if scoped?
- Did ASR work if scoped?
- Did character tags work?
- Did pause tags work?
- Did parameter switching work?
- Did Clear VRAM then regenerate work?
- Did interrupt/cancel work in long generation?
```

## Important Failures This Prevents

This checklist exists because working audio is not enough.

Common failures:

- Engine works once, then fails after ComfyUI unloads models.
- SRT generation cannot be cancelled.
- Cache is missing, so repeated generations waste time and VRAM.
- UI exposes fake language/speed/emotion controls.
- Reference audio uses the wrong sample rate and changes pitch.
- Character tags work in TTS Text but not SRT.
- Pause tags break because segments are handled incorrectly.
- Model files download to a hidden cache instead of `ComfyUI/models/TTS/`.
- Unified node gets filled with engine-specific code.

If any item fails, fix it before PR review.

---

Navigation: [Guide Hub](README.md) | Previous: [04 - Implementation Order For LLM](04_IMPLEMENTATION_ORDER_FOR_LLM.md) | Next: [06 - User Prompts To Copy Paste](06_USER_PROMPTS_TO_COPY_PASTE.md)
