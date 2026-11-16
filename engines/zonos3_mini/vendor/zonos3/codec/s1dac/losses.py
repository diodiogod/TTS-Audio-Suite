"""Mimi-style SSL frame averaging and the frozen W2V-BERT semantic teacher (the inference parts of the s1dac losses)."""

import torch
from torch import nn
from torch.nn import functional as F


def average_ssl_frames(target, student_frames, window_ratio=2.0):
    """Noncausal box averages, including all overlapping SSL frame cells.

    Mimi uses kernel=8, stride=4 for 50 -> 12.5 Hz. Here a student interval
    occupies target_frames/student_frames SSL frame cells, and the window is
    window_ratio intervals wide, centered on that interval. With ratio 4 and
    window_ratio 2 this is avg_pool1d(kernel=8,stride=4,padding=2,
    count_include_pad=False).

    Fractional boundary cells contribute in proportion to their overlap.
    Integral prefix sums implement exact piecewise-constant box integration
    without interpolating feature vectors or rounding away any teacher frames.
    Crop-edge windows are clipped and renormalized over available frames.
    """
    if target.ndim != 3 or target.shape[1] < 1 or student_frames < 1:
        raise ValueError("Expected nonempty SSL features [batch,frames,channels]")
    if window_ratio <= 0:
        raise ValueError("window_ratio must be positive")
    with torch.autocast(target.device.type, enabled=False):
        values = target.detach().float().transpose(1, 2)
        teacher_frames = values.shape[-1]
        ratio = teacher_frames / student_frames
        centers = (torch.arange(student_frames, device=target.device, dtype=torch.float32) + 0.5) * ratio
        half_width = 0.5 * window_ratio * ratio
        left = (centers - half_width).clamp(0, teacher_frames)
        right = (centers + half_width).clamp(0, teacher_frames)
        prefix = F.pad(values.cumsum(dim=-1), (1, 0))

        def integral(position):
            index = position.floor().long()
            fraction = position - index
            return (
                prefix.index_select(-1, index) + values.index_select(-1, index.clamp(max=teacher_frames - 1)) * fraction
            )

        return (integral(right) - integral(left)) / (right - left)


class SemanticTeacher(nn.Module):
    def __init__(self, path, layer=16, dtype=torch.bfloat16):
        super().__init__()
        if layer != 16:
            raise ValueError("The recipe targets transformer layer 16")
        from transformers import Wav2Vec2BertModel

        self.model = Wav2Vec2BertModel.from_pretrained(path, dtype=dtype, local_files_only=True)
        self.model.encoder.layers = nn.ModuleList(list(self.model.encoder.layers[:layer]))
        self.model.config.num_hidden_layers = layer
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, features, attention_mask):
        self.model.eval()
        return self.model(
            input_features=features.to(next(self.model.parameters()).dtype),
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
