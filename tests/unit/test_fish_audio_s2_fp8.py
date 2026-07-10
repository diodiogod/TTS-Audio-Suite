from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.runtimes.fish_audio_s2_fp8 import FP8Linear


@pytest.mark.unit
def test_fp8_linear_uses_row_scales_for_bfloat16_compute():
    layer = FP8Linear(2, 2)
    layer.qweight = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float8_e4m3fn,
    )
    layer.scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    inputs = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)

    output = layer(inputs)

    expected_weight = torch.tensor(
        [[0.5, 1.0], [0.75, 1.0]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(output, torch.nn.functional.linear(inputs, expected_weight))
