import logging

import torch

from funasr_detach.models.transformer.model import Transformer
from funasr_detach.register import tables


@tables.register("model_classes", "Conformer")
class Conformer(Transformer):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
