from __future__ import annotations

import torch
import torch.nn.functional as F

from zonos3_sgl.layers import BaseOP


class SpeakerLDAProjection(BaseOP):
    """Affine LDA projection applied to raw speaker embeddings.

    The weights are stored in the model checkpoint as
    ``speaker_lda_projection.weight`` and ``speaker_lda_projection.bias``.
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.weight = torch.empty(int(output_dim), int(input_dim))
        self.bias = torch.empty(int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
