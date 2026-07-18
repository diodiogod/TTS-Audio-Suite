"""Compatibility helpers for Step Audio EditX's Transformers inference backend."""

from __future__ import annotations

import math
import types
import torch
import torch.nn.functional as F


STEP_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ '<s>' }}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{% set role = 'human' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<|BOT|> ' + role + '\\n' }}"
    "{{ message['content'] }}"
    "{% if not loop.last or message['role'] != 'assistant' %}{{ '<|EOT|>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|BOT|> assistant\\n' }}{% endif %}"
)

STEP_AUDIO_TOKEN_START = 65536
STEP_VQ02_TOKEN_END = STEP_AUDIO_TOKEN_START + 1024
STEP_VQ06_TOKEN_START = STEP_VQ02_TOKEN_END
STEP_AUDIO_TOKEN_END = STEP_AUDIO_TOKEN_START + 5120


def ensure_chat_template(tokenizer):
    """Supply the current official Step template to older model snapshots."""
    # TTS Audio Suite patch: early Step snapshots omitted chat_template.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = STEP_CHAT_TEMPLATE
    return tokenizer


def has_optimus_attention() -> bool:
    """Return whether Step's optional proprietary attention op is available."""
    try:
        return callable(torch.ops.Optimus.fwd)
    except (AttributeError, RuntimeError):
        return False


def _alibi_slopes(num_heads: int, *, device: torch.device) -> torch.Tensor:
    """Reproduce Step's ALiBi slopes without allocating an attention square."""
    nearest_power_of_two = 2 ** math.floor(math.log2(num_heads))
    base = 2.0 ** (-8.0 / nearest_power_of_two)
    slopes = torch.pow(
        torch.tensor(base, device=device),
        torch.arange(1, nearest_power_of_two + 1, device=device),
    )
    if nearest_power_of_two < num_heads:
        extra_base = 2.0 ** (-4.0 / nearest_power_of_two)
        extras = torch.pow(
            torch.tensor(extra_base, device=device),
            torch.arange(
                1,
                1 + 2 * (num_heads - nearest_power_of_two),
                2,
                device=device,
            ),
        )
        slopes = torch.cat((slopes, extras))
    return slopes


def build_rectangular_alibi_mask(
    query_positions: torch.Tensor,
    key_length: int,
    num_heads: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build only the ALiBi rows required by the current generation step."""
    query_positions = query_positions.to(device=device, dtype=torch.long).reshape(-1)
    key_positions = torch.arange(key_length, device=device)
    distances = query_positions[:, None] - key_positions[None, :]
    causal = distances >= 0
    bias = -torch.sqrt(distances.clamp_min(0).to(torch.float32))
    slopes = _alibi_slopes(num_heads, device=device)
    bias = slopes[:, None, None] * bias[None, :, :]
    bias = bias.masked_fill(~causal[None, :, :], float("-inf"))
    return bias.to(dtype=dtype)


def _efficient_sdpa_forward(
    self,
    x: torch.Tensor,
    past_key_value=None,
    attention_mask: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
):
    """Memory-safe replacement for StepAttention's missing-Optimus fallback."""
    del attention_mask  # Step's original implementation also replaces this mask.

    batch_size, query_length, _ = x.shape
    query = self.q_proj(x)
    key = self.k_proj(x)
    value = self.v_proj(x)

    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        key, value = past_key_value.update(
            key,
            value,
            self.layer_idx,
            cache_kwargs,
        )

    key_length = key.shape[1]
    query = query.view(
        batch_size,
        query_length,
        self.num_heads,
        self.head_dim,
    )
    key = key.view(
        batch_size,
        key_length,
        self.num_groups,
        self.head_dim,
    )
    value = value.view(
        batch_size,
        key_length,
        self.num_groups,
        self.head_dim,
    )
    key = key.repeat_interleave(self.num_heads // self.num_groups, dim=2)
    value = value.repeat_interleave(self.num_heads // self.num_groups, dim=2)

    if cache_position is not None and cache_position.numel() == query_length:
        query_positions = cache_position
    else:
        query_positions = torch.arange(
            key_length - query_length,
            key_length,
            device=x.device,
        )

    alibi_mask = build_rectangular_alibi_mask(
        query_positions,
        key_length,
        self.num_heads,
        dtype=query.dtype,
        device=query.device,
    )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=alibi_mask,
        dropout_p=0.0,
    )
    output = output.transpose(1, 2).flatten(-2, -1)
    return self.o_proj(output), None


def install_memory_safe_attention(model) -> str:
    """Install the SDPA fallback when Step's Optimus op is unavailable."""
    if has_optimus_attention():
        return "optimus"

    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "StepAttention":
            continue
        module.forward = types.MethodType(_efficient_sdpa_forward, module)
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "Step Audio EditX compatibility error: no StepAttention layers found."
        )
    return f"pytorch_sdpa ({patched} layers)"
