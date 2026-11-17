# Vendored ZONOS3-Mini implementations

- Source: https://github.com/Zyphra/ZONOS3-Mini
- Commit: `96d9e0ec1e39083ed336ac78f24e3dbc93721c0f`
- `zonos3/` <- `model/src/zonos3/`: the reference implementation (import closure of `zonos3.reference.synth` plus
  `models/text_extra.py` and the three modules the fast engine adds: `models/config.py`, `utils/hf.py`,
  `utils/moe_topk.py`), minus the modules listed under "Not included".
- `zonos3_sgl/` <- `servers/sgl/src/zonos3_sgl/`: the CUDA-graph engine, reduced to what one in-process scheduler
  needs and with its compiled kernels replaced; see "Fast engine".
- License: MIT (`LICENSE`, `NOTICE`, `third_party/licenses/`). Model weights (`Zyphra/ZONOS3-Mini` on Hugging Face) are
  Apache-2.0 and are downloaded separately into `ComfyUI/models/TTS/zonos3_mini/`.
- `../assets/` holds the parts of upstream `assets/` that the reference implementation reads; see "Assets".

The packages keep their absolute `zonos3.*` / `zonos3_sgl.*` imports; `engines/zonos3_mini/__init__.py::ensure_vendor_path()` puts this
folder on `sys.path`.

## Patches to `zonos3/`

Every change is marked `# TTS Audio Suite Patch:` in the source. All other Python files and the registries are
byte-identical to the commit.

1. `zonos3/utils/release.py` — `resolve_file` / `resolve_dir_or_local` map `hf://Zyphra/ZONOS3-Mini/...` into a local
   model root set with `set_local_root()`; the `huggingface_hub` downloads, `_token` and the fetch prints are removed.
2. `zonos3/tokenizer/matched_codec.py` — decoder head checkpoint loaded with `weights_only=True` (was `False`).
3. `zonos3/models/speaker_cloning.py` — legacy `Qwen3SpeakerEmbedding`, `_AttentiveStatisticsPooling`,
   `W2VBertSVSpeakerEmbedding`, their `torchaudio` / `huggingface_hub` / `transformers` / `zonos` imports,
   `DEFAULT_DEVICE`, the alias table and the 2048D/256D fallbacks are removed; only the registry encoder remains.
4. `zonos3/codec/data_processing/torch_utils.py` — `import fcntl` moved into the inductor branch that uses it, so the
   module imports on Windows.
5. `zonos3/models/third_party/antspeaker/models/model.py` — `torch.load(..., weights_only=True)`.

Removed `if __name__ == "__main__":` demo blocks: `zonos3/codec/s1dac/vendor/modded_dac.py`,
`zonos3/codec/s1dac/vendor/rvq.py`, `zonos3/tts/laci.py`.

## Not included

Modules of the import closure that are only imported lazily, on paths this integration never takes:

- `zonos3/codec/istft_head_a_20261002/stream.py`, `zonos3/codec/stream_device.py` — streaming decode. The engine
  always requests `stream_decode: "whole"`; `matched_codec.stream()` / `capture_stream_graphs()` and the `on_audio`
  branch of `synthesize()` would need them.
- `zonos3/tts/edit.py` — speech editing, which needs model exports that are not published.

## Fast engine (`zonos3_sgl/`)

Upstream's engine compiles C++/CUDA at start-up (`tvm_ffi`, needing `nvcc` and a host compiler) and imports
`sgl_kernel`, `flashinfer`, `fla-core`, ZeroMQ and MessagePack. This copy needs PyTorch and Triton only.
`engines/zonos3_mini/zonos3_mini_fast.py` drives the scheduler in-process. Changed files carry
`# TTS Audio Suite Patch:`; the rest are byte-identical to the commit.

Kernels replaced with PyTorch ops (token output stayed identical to upstream's native build):

- `kernel/index.py`, `kernel/store.py`, `kernel/__init__.py` — embedding gather and KV store (were JIT-compiled CUDA).
- `layers/moe/fused_moe/fused_moe_impl.py`, `layers/moe/fused_moe/topk.py`, `layers/activation.py` — MoE block
  alignment, top-k and `silu_and_mul` (were `sgl_kernel` / `flashinfer`).
- `layers/norm.py`, `layers/rotary.py` — RMSNorm and RoPE: only the PyTorch ("training") variants remain.

Kernels replaced with new Triton kernels (not upstream files; these round differently, so tokens differ from
upstream's native build for the same seed):

- `kernel/triton/paged_attention.py` — causal attention over the paged KV cache, replacing FlashAttention-3
  (`sgl_kernel.flash_attn`); called from `attention/fa.py`.
- `kernel/triton/gated_delta.py` and a rewritten `models/gdn.py` — the gated-delta layers run the reference
  implementation's per-token recurrence instead of `fla-core`'s 64-frame chunked kernels. A slot's state is always at
  the request's last row, so there is no per-step replay of the unfinished chunk and no FLA cache pinning
  (`engine/engine.py`, `engine/prefill_graph.py`).

Removed:

- Server and transport: `server/`, `tokenizer/`, `shell.py`, `__main__.py`, `benchmark/`, `utils/mp.py`, the
  ZeroMQ queues in `scheduler/io.py`, NCCL (`kernel/pynccl.py`, `distributed/impl.py`), `kernel/csrc/`.
- The offline API `tts/llm.py` and `tts/sequence.py` (replaced by `zonos3_mini_fast.py`).
- Unused back ends: `attention/fi.py` (flashinfer), `kvcache/radix_manager.py`, `kernel/radix.py`,
  `kernel/tensor.py`, `kernel/utils.py`, `layers/moe/moe_backend.py`, `utils/arch.py`, the `tqdm` progress bar in
  `engine/graph.py`.
- Accent adapters: `models/accent_adapters.py`, `kernel/triton/expert_lora.py`; `EngineConfig.accent_adapters`
  raises. The release checkpoint ships none.

## Assets

Copied from upstream `assets/`, reduced to what the loaders read. Vectors (`.npy`) are byte-identical.

- `steering/`: the bundle `studio_bundle_mect60k_x10.json` and the 14 direction sets it names. The other five set
  folders and every `spec.json` / `report.json` / evaluation file are left out. The bundle keeps `name`, `sets`,
  `timing`, `rate_extension`, `steer_release` (the studio's `ui` table is dropped); each `manifest.json` keeps `dim`,
  `space`, `source_index` and per direction `file`, `kind`.
- `laci_heads/`: each file keeps `frames_per_token` and per head `layer`, `head`, `radius`, the fields
  `LaciHeads.load` reads; the selection statistics are dropped.
- `default_voices/`: the `.codes.npy` / `.embedding.npy` sidecars only (no audio clips, no README).

To refresh from upstream, copy the folders again and re-apply the same field selection.
