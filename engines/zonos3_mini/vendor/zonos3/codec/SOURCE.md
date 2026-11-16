# Provenance of `zonos3.codec`

Vendored from the s1_s2_dac codec package of the release codec `opus_dual_3n_600k_sched600k_r4096_step600000`.

- `P` = `/data/groups/audio/gabriel/s1_s2_dac/downstream/campaigns/opus_dual_finished_50k_20260912/packages/3n_600k_sched600k_r4096/source`
- `R` = `/data/groups/audio/gabriel/s1_s2_dac`

| File | Source |
| --- | --- |
| `codec_adapter.py` | bingua2 `minisgl/vendor/dual_codec_adapter/codec_adapter.py` (f543d63) |
| `resample_exact.py` | `P/resample_exact.py` |
| `istft_head_a_20261002/{__init__,model}.py` | `R/istft_head_a_20261002/` |
| `hybrid10_dual_encoding/{__init__,base_model,inference,model}.py` | `P/hybrid10_dual_encoding/` |
| `s1dac/{__init__,losses}.py` | `P/s1dac/` |
| `s1dac/vendor/{__init__,modded_dac,rvq}.py` | `P/s1dac/vendor/` |
| `s1dac/same_data/npy_ogg_dataset.py` | `P/s1dac/same_data/npy_ogg_dataset.py` L48-51 (`_cached_resampler`) and L85-118 (`Wav2Vec2BertFeaturePreprocessor`), verbatim |
| `data_processing/{independent_batcher,torch_utils}.py` | `P/data_processing/` |

## Changes

- Every import of `data_processing`, `resample_exact`, `hybrid10_dual_encoding` and `s1dac` is rewritten to
  `zonos3.codec.*`; the runtime no longer puts codec directories on `sys.path`.
- `s1dac/same_data/npy_ogg_dataset.py` keeps only the W2V-BERT preprocessor; the torchcodec/ctypes dataset code is not
  used at inference. `s1dac/same_data/__init__.py` is empty.
- `data_processing/__init__.py` holds a docstring only (no `raw_audio_dataloader` import, no `__getattr__`).
- `s1dac/losses.py` keeps only `SemanticTeacher` and `average_ssl_frames` (the training losses are not used at
  inference).
- `s1dac/vendor/modded_dac.py` `DecoderBlock` no longer builds the transformer it never used (its `nn.Sequential`
  always had it commented out).
- `codec_adapter.py` (`MossAutoencoder`):
  - takes the generator checkpoint, SSL teacher dir, step, codebook sizes, codec name, checkpoint sha256 and device as
    keyword arguments (`tokenizer/dual_reference.py` fills them from the codec registry) instead of reading the
    package's `checkpoints/hybrid_10cb.json` receipt; the device is no longer hardcoded to `cuda:0`;
  - drops `ROOT`, `add_arguments` and `create_autoencoder`;
  - drops the real-audio validation on `audio_examples.pt` (not released) and the `checks/` report file; the random
    warm-up graph/native agreement checks stay;
  - `metadata()` reports `ssl_teacher` (the teacher dir) instead of the package's `teacher.sha256` manifest.
