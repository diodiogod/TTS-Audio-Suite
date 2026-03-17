# Auxiliary Model Sources

Helper/post-process models used by standalone utility nodes.
These are not engines and are documented separately to avoid polluting the engine registry.

## Punctuation / Truecase

Optional punctuation and true-casing post-process models for raw ASR text. Designed mainly for low-punctuation transcripts such as Granite outputs, but usable on any plain text.

**Used by:** 📝 ASR Punctuation / Truecase

| Model | Repo | Size | Auto-Download | Languages | Notes |
|---|---|---|---|---|---|
| English Fullstop + Truecase | [1-800-BAD-CODE/punctuation_fullstop_truecase_english](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english) | ~210MB | ✅ | English | English-only. Best on lowercased, unpunctuated ASR text. |
| Romance Fullstop + Truecase | [1-800-BAD-CODE/punctuation_fullstop_truecase_romance](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_romance) | ~144MB | ✅ | Spanish, French, Italian, Portuguese, Catalan, Galician | Romance-language punctuation and true-casing model. |
| 47-Language Fullstop + Truecase | [1-800-BAD-CODE/punct_cap_seg_47_language](https://huggingface.co/1-800-BAD-CODE/punct_cap_seg_47_language) | ~1.11GB | ✅ | 47 languages | Heavier multilingual fallback with the widest language coverage supported by the punctuators package. |

*Generated from [tts_audio_suite_aux_models.yaml](Dev%20reports/tts_audio_suite_aux_models.yaml).*
