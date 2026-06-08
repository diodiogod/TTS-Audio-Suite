TTS Audio Suite - Transformers 5.x Higgs Audio 2 Investigation Report

Scope
- Problem: Higgs Audio 2 regressed after the suite moved the main environment to Transformers 5.x.
- Goal: determine whether Higgs Audio 2 is a good native Transformers 5 patch target, or whether it should move to runtime isolation.

Environment
- Windows / ComfyUI runtime.
- Main env on Transformers 5.x.
- Direct repro and validation performed in the real Windows Python environment outside the ComfyUI UI wrapper.

Findings Summary
- Higgs Audio 2 is not blocked at the shallow import/API level only.
- Some real compatibility fixes were possible:
  - bundled cache API adaptation
  - cache-arg rename fixes
  - major cold-load regression fix
- However, the remaining failure is decode-path semantics, especially around StaticCache and CUDA graph replay.
- Higgs can be made to load and "run" on Transformers 5, but the high-performance path is not trustworthy and produced incorrect audio.
- This is not a good short-path patch target like IndexTTS.

Reproduction
1. Transformers 5 main environment.
2. Higgs Audio 2 loaded through the normal TTS Text path with voice cloning.
3. Observed one or more of:
   - multi-minute cold-load stalls
   - StaticCache API crashes
   - CUDA graph capture/runtime failures
   - runaway/gibberish generation
   - near-empty or silent output

Key Diagnostic Evidence
- Initial cold-load timing showed:
  - `⏱️ Higgs Audio: Core model load took 440.61s`
- Tokenizer/HuBERT stages were not the bottleneck:
  - text tokenizer around 1-2s
  - audio tokenizer around 2-5s
- The 440s stall was inside `HiggsAudioModel.from_pretrained(...)`, specifically Hugging Face missing-key initialization.
- Direct checkpoint/model key comparison in runtime showed:
  - `missing_count 0`
  - `unexpected_count 0`
- After bypassing unnecessary missing-key initialization for complete local checkpoints, direct timing dropped to:
  - `⏱️ Higgs Audio: Core model load took 58.46s`
  - later direct runs around `33-58s` depending on cache state

- StaticCache/API drift evidence:
  - `'StaticCache' object is not subscriptable`
  - later mask/cache shape mismatches around static cache width `1024`

- Cache-arg drift evidence:
  - bundled Higgs/LLaMA path mixed `past_key_value` and `past_key_values`
  - led to duplicated kwarg failures and broken decode semantics

- CUDA graph replay evidence:
  - direct runtime probe with graph replay active produced:
    - immediate or near-immediate audio termination
    - tiny garbage clips
    - or invalid decode behavior
  - direct runtime probe with StaticCache but without CUDA graph replay produced:
    - normal-length audio token generation
    - reasonable token spread
    - real waveform output

Attempts and Results (chronological)
1) StaticCache compatibility shims
   - Added bundled helpers to access cache tensors/device across old tuple caches and new Cache/StaticCache APIs.
   - Result: fixed immediate StaticCache indexing crashes.

2) CUDA graph failure fallback cleanup
   - On graph capture failure, forced cache recreation and switched to DynamicCache instead of continuing with stale StaticCache assumptions.
   - Result: removed some follow-on crashes after graph failure, but not the core correctness issue.

3) Attention backend adjustment
   - Stopped forcing eager attention on CUDA and preferred SDPA instead.
   - Result: better alignment with the intended high-performance path, but did not fully restore correctness.

4) TTS defaults forcing audio generation
   - Set Higgs TTS defaults to force audio generation rather than letting the model drift in text mode.
   - Result: prevented obvious free-running text-style generation, but did not solve bad output.

5) Cold-load regression fix
   - Added a local complete-checkpoint detector and skipped Hugging Face missing-key initialization when the local Higgs checkpoint was complete.
   - Result: large real improvement to cold-load time.

6) Cache-argument adaptation
   - Patched bundled Higgs/LLaMA call sites for Transformers 5 cache-arg naming.
   - Result: moved runtime further, but surfaced deeper semantic issues instead of fixing them outright.

7) CUDA graph runner PyTorch compatibility
   - Wrapped input-buffer updates in inference mode to satisfy current PyTorch restrictions.
   - Result: removed one explicit runtime crash, but not the underlying decode corruption.

8) StaticCache without CUDA graph replay direct comparison
   - Direct Windows probes showed:
     - `enable_cuda_graphs=True` with graph replay: corrupted/too-short audio output
     - `enable_cuda_graphs=False`: long audio-token generation and real waveform output
     - `StaticCache` with graph replay skipped: closer to sane behavior than replay path
   - Result: isolated the remaining failure to the CUDA graph replay path on Transformers 5.

Interpretation
- Higgs Audio 2 on Transformers 5 is not blocked by a single missing symbol or a few renamed APIs.
- The remaining problem is decode/runtime behavior, especially the interaction between:
  - StaticCache
  - bundled Higgs decode logic
  - LLaMA attention/cache semantics on current Transformers 5
  - CUDA graph replay
- That is behavior-level drift, not just compatibility-shim work.

Decision
- Higgs Audio 2 on Transformers 5 is still theoretically patchable.
- It is not a good use of time right now.
- The engine reached the same bad zone as Qwen3-TTS, but with even more sensitivity around CUDA graphs and cache behavior.
- Runtime isolation is the correct path for now.

Operational Recommendation
- Do not continue native Transformers 5 Higgs patching in the main branch.
- Route Higgs Audio 2 into an isolated legacy runtime instead.
- Prefer a dedicated Higgs runtime first rather than assuming shared legacy parity.
- Revisit native Transformers 5 support only if Higgs becomes important enough to justify deeper decode/cuda-graph compatibility work.

Notes
- The cold-load fix was a real improvement and may still be worth revisiting later, even outside the final T5 decision.
- The rest of this investigation should be treated as a branch-local investigation record, not as a stable main-branch solution.
