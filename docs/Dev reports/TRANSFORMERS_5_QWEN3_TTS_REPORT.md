TTS Audio Suite - Transformers 5.0.0 Qwen3-TTS Investigation Report

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
