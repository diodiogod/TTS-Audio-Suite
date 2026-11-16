# Vendored ZONOS3-Mini reference implementation

- Source: https://github.com/Zyphra/ZONOS3-Mini
- Commit: `96d9e0ec1e39083ed336ac78f24e3dbc93721c0f`
- Upstream path: `model/src/zonos3/` -> `zonos3/` (import closure of `zonos3.reference.synth` plus
  `models/text_extra.py`; the SGL/CUDA-kernel server `zonos3_sgl` is not included)
- License: MIT (`LICENSE`, `NOTICE`, `third_party/licenses/`). Model weights (`Zyphra/ZONOS3-Mini` on Hugging Face) are
  Apache-2.0 and are downloaded separately into `ComfyUI/models/TTS/zonos3_mini/`.
- `../assets/` holds upstream `assets/steering`, `assets/laci_heads` and the `.codes.npy` / `.embedding.npy` sidecars
  of `assets/default_voices` (no audio clips).

The package keeps its absolute `zonos3.*` imports; `engines/zonos3_mini/__init__.py::ensure_vendor_path()` puts this
folder on `sys.path`.

## Patches

Every change is marked `# TTS Audio Suite Patch:` in the source. All other files are byte-identical to the commit.

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
