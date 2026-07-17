import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class Qwen3TextEncoder(nn.Module):
    """Wraps Qwen3 (decoder-only) as a text encoder for Wan audio pipeline.

    Loads the full Qwen3 model and extracts last-layer hidden states
    as text embeddings. Interface matches WanTextEncoder.forward(ids, mask).
    """

    def __init__(self, model_path, torch_dtype=torch.bfloat16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # TTS Audio Suite patch: Transformers 5 renamed torch_dtype to dtype.
            # Hidden states are requested during forward instead of being stored
            # as an invalid generation-config flag while loading the model.
            dtype=torch_dtype,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.dim = self.model.config.hidden_size  # 2048 for Qwen3-1.7B

    def forward(self, ids, mask=None):
        """
        Args:
            ids:  [batch, seq_len] token ids
            mask: [batch, seq_len] attention mask (1=valid, 0=pad)
        Returns:
            hidden_states: [batch, seq_len, dim] last-layer hidden states
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
        return outputs.hidden_states[-1]
