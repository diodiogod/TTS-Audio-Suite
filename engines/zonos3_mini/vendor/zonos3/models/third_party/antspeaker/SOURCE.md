# Vendored AntSpeaker model code

`antspeaker/` = model code from https://github.com/ant-research/AntSpeaker at commit `41dcd09`: `models/{__init__,model,poolings}.py`,
`models/backbones/{__init__,mect,moe}.py`, `utils/registry.py`, taken from George's `w2v-bert-sv-mect` recipe
(`recipes/DeepASV/local/third_party/antspeaker`, copied there unmodified on 2026-10-06) on 2026-10-16.

The only change here: the absolute import prefix `antspeaker.` is rewritten to `zonos3.models.third_party.antspeaker.`
so the package imports without a sys.path hack. Every `.py` file keeps its "Apache License, Version 2.0" header
(Copyright (c) 2026 Ant Group Co., Ltd.). The upstream repository's LICENSE / README state CC-BY-NC-SA 4.0 for the
released checkpoints (`mect_*.pt`): those are never loaded here. The weights served through this code are George's
own-weights MECT-B2 trained from scratch on our data (`ownmect_s2`, see `speaker_encoder_registry.json`).

Used by `zonos3/models/mect_speaker.py` (MECTFbankSpeakerEmbedding) for checkpoints whose
`speaker_embedding_codec` names a registry entry of type `mect_fbank`.
