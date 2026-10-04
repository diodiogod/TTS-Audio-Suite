import logging

from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1



def drop_invalid_tokens(x):
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    if SOS in x:
        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
    else:
        s = 0

    if EOS in x:
        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
    else:
        e = None

    x = x[s: e]

    # T3's speech head spans `speech_tokens_dict_size` (8194) while S3Gen's
    # input embedding only has SPEECH_VOCAB_SIZE (6561) rows, so a sampled id
    # above the vocab that is not exactly SoS/EoS reaches the embedding gather
    # and trips a device-side "index out of bounds" assert (which surfaces
    # later, at the next CUDA sync, with a misleading traceback).
    invalid = x >= SPEECH_VOCAB_SIZE
    if invalid.any():
        logging.warning(
            "Dropping %d out-of-vocabulary speech token(s) (max id %d, vocab %d)",
            int(invalid.sum()), int(x.max()), SPEECH_VOCAB_SIZE,
        )
        x = x[~invalid]
    return x
