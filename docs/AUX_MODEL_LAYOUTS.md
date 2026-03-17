# Auxiliary Model Layouts

Use this document for helper/post-process model folder structures and placement notes.
These are not engine weights. They are optional models used by standalone utility nodes.

For repository/source URLs, use [AUX_MODEL_SOURCES.md](AUX_MODEL_SOURCES.md).

## Punctuation / Truecase

```text
ComfyUI/models/TTS/punctuation/
├── punctuation_fullstop_truecase_english/
├── punctuation_fullstop_truecase_romance/
└── punct_cap_seg_47_language/
```

Notes:

- Models download lazily on first use by the punctuation node.
- These are helper models for text cleanup, not TTS/ASR engines.
- The English model is the lightest and is mainly intended for raw English ASR text.
- The 47-language model is much heavier and should be treated as the multilingual fallback.
