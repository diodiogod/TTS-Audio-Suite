TTS Audio Suite - Transformers 5.x Qwen3-TTS Investigation Report

Scope
- Problem: Qwen3-TTS garbled audio or load failure on Transformers 5.0.0.
- Goal: identify a safe compatibility path or conclude unsupported for now.

Findings Summary
- The tokenizer path in Transformers 5.0.0 is the core blocker.
- AutoProcessor/AutoTokenizer paths either fail or apply an incorrect regex fix and lead to broken token IDs.
- Direct Qwen2Tokenizer loading via vocab/merges is broken in this environment (encoder remains empty or vocab_size=1).
- The fast tokenizer build using tokenizers BPE can be created in isolation, but in ComfyUI runtime PreTrainedTokenizerFast fails to load vocab from a temporary directory (error: unable to load vocabulary).
- The model config's eos/pad tokens (<|im_end|>, <|endoftext|>) are not present in vocab.json; safe fallback tokens that exist are <s>, <unk>, </s>.

Reproduction (Windows / ComfyUI)
1) Transformers 5.0.0 environment.
2) Qwen3-TTS model folder present with vocab.json and merges.txt.
3) Qwen3-TTS load fails or produces garbled output.

Key Diagnostic Evidence
- Fast tokenizer build using tokenizers (BPE) works in an isolated test:
  - vocab_size=151643
  - bos=<s> (id 128245), unk=<unk> (id 128244), eos/pad=</s> (id 128247)
- In ComfyUI runtime, PreTrainedTokenizerFast.from_pretrained(tmpdir) fails with:
  - "Unable to load vocabulary from file"
- The tokenizer_config.json tokens (<|im_end|>, <|endoftext|>) do not exist in vocab.json.

Attempts and Results (chronological)
1) Rope fallback normalization for transformers 5.0.0
   - Added resolve_rope_type_with_fallback and patched rope_scaling.
   - Reduced KeyError crashes but did not fix garbled audio.
2) AutoProcessor / AutoTokenizer (fix_mistral_regex)
   - On 5.0.0, fix_mistral_regex triggers duplicate-arg errors or incorrect regex warning.
   - Still leads to bad tokenization.
3) Direct Qwen2Tokenizer(vocab/merges)
   - In runtime, encoder remains empty; vocab_size collapses to 1.
   - Garbled output even when forced ids are set.
4) Fast tokenizer BPE build using tokenizers (Tokenizer + BPE)
   - Works in isolated script; IDs look correct.
   - Fails in runtime when wrapped by PreTrainedTokenizerFast (from_pretrained tmpdir).
5) Token ID correction
   - For tokens not present in vocab.json, fallback to <s>/<unk>/<\s>.
   - Valid in test but runtime build still blocked by tokenizer load error.

Conclusion
- With Transformers 5.0.0 as installed in the reported environment, the tokenizer path is blocked.
- This is not fixed by rope patches or token id corrections alone.
- A working solution likely requires either:
  - a Transformers build that accepts tokenizer_object (or can load tokenizer.json without requiring vocab/merges), or
  - downgrading Transformers to a 4.x release known to work with Qwen3-TTS.

Recommendation
- Temporarily mark Transformers 5.0.0 as unsupported for Qwen3-TTS and emit a clear warning.
- Provide guidance to downgrade Transformers or use a compatible build.

Notes
- No changes in this report alter project behavior.

Addendum - Transformers 5.10.2 Re-check

Scope
- Problem: determine whether newer Transformers 5.10.2 improves Qwen3-TTS enough to avoid runtime isolation.
- Goal: distinguish shallow API drift from deeper generation/runtime incompatibility.

Environment
- Windows / ComfyUI runtime.
- Main env on Transformers 5.10.2.
- Direct repro also performed in the real Windows Python environment outside the ComfyUI UI wrapper.

What Changed vs 5.0.0
- The failure mode is different.
- The first blocker is no longer the tokenizer collapse described above.
- Qwen3-TTS can be pushed significantly further on 5.10.2 with local compatibility shims.

Key Diagnostic Evidence
- Direct model load initially failed with:
  - `KeyError: 'default'`
  - source: `ROPE_INIT_FUNCTIONS[self.rope_type]`
- Current Transformers 5.10.2 no longer exposes a legacy `"default"` RoPE entry:
  - available keys observed in runtime: `dynamic`, `linear`, `llama3`, `longrope`, `proportional`, `yarn`
- After restoring a local `"default"` compatibility entry, model load succeeded.
- Generation then failed on changed Hugging Face masking APIs:
  - `create_causal_mask() got an unexpected keyword argument 'input_embeds'`
- After adapting those call sites to the current HF signature, generation progressed further.
- Final direct generation failure on 5.10.2 was:
  - `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [16, 277] but got: [16, 139].`
  - source: SDPA attention path during generation.

Attempts and Results (5.10.2)
1) Legacy RoPE compatibility restore
   - Added a local fallback for legacy `"default"` RoPE initialization.
   - Result: Qwen3-TTS load progressed and completed successfully.
2) Mask helper API adaptation
   - Updated bundled Qwen mask call sites to match current HF `create_causal_mask` / `create_sliding_window_causal_mask` signatures.
   - Result: generation progressed beyond the earlier immediate crash.
3) Direct generation validation with real voice-clone inputs
   - Used real reference audio and transcript in the Windows environment.
   - Result: no longer a trivial import/load/signature issue; generation hit a tensor-shape mismatch inside SDPA.

Interpretation
- Transformers 5.10.2 is not blocked in the same way as 5.0.0.
- Some problems are still shallow compatibility issues and can be adapted cleanly.
- However, once generation reaches SDPA tensor-shape mismatch territory, this stops being a simple API-shim problem.
- At that point, continuing the patch path means re-validating model behavior against newer Hugging Face generation/mask/cache semantics, not just restoring removed symbols.

Decision
- Qwen3-TTS on Transformers 5.10.2 is still technically patchable in principle.
- It is not a good short-path target like IndexTTS.
- The suite should prefer runtime isolation for Qwen3-TTS instead of continuing deep Transformers 5 surgery.

Operational Recommendation
- Treat Qwen3-TTS as a legacy-stack engine for now.
- Route it into the shared legacy Transformers 4 runtime used by compatible engines (same family already proven for VibeVoice/Qwen in prior testing).
- Keep the main environment on Transformers 5.
- Revisit native Transformers 5 support only if Qwen3-TTS becomes strategically important enough to justify deeper behavior-level compatibility work.

Notes
- The transient 5.10.2 patch attempts were not kept as the main solution path.
- The current project direction is runtime isolation, not continued in-place Transformers 5 patching for Qwen3-TTS.
