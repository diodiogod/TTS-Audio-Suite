# Text Chunking Guide

TTS Audio Suite supports long text generation through the shared unified chunking system.

## What It Does

- splits long text into smaller generation chunks
- preserves sentence boundaries when possible
- can fall back to comma-based splitting for very long sentences
- combines generated audio using the suite chunk combiner

## Main Controls

- `enable_chunking`
  - turns chunking on or off
- `max_chars_per_chunk`
  - character limit per chunk
- `chunk_combination_method`
  - `auto`
  - `concatenate`
  - `silence_padding`
  - `crossfade`
- `silence_between_chunks_ms`
  - silence inserted when the selected combination method uses silence padding

## Auto Combination Behavior

When `chunk_combination_method=auto`, the suite picks a join strategy based on text size:

- longer text tends toward `silence_padding`
- medium text tends toward `crossfade`
- shorter chunked text tends toward `concatenate`

The exact choice is made by the shared chunk combiner, not by engine-local custom logic.

## Important Architecture Rule

Chunking is a suite feature.

- new engines should reuse the shared chunking and chunk-combination controls
- engines should not invent duplicate local chunk-silence or chunk-join settings unless the official model has a real distinct native long-form system

## Related Docs

- [Example Workflows](../README.md#-example-workflows)
- [Character Switching Guide](./CHARACTER_SWITCHING_GUIDE.md)
- [Multiline TTS Tag Editor Guide](./MULTILINE_TTS_TAG_EDITOR_GUIDE.md)
