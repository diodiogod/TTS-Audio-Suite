"""Text tokenizer for KugelAudio based on Qwen2."""

from typing import List, Optional

from transformers.utils import logging
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

logger = logging.get_logger(__name__)


class KugelAudioTextTokenizer(Qwen2TokenizerFast):
    """Text tokenizer for KugelAudio with speech special tokens.
    
    Based on Qwen2 tokenizer with additional tokens for speech synthesis:
    - speech_start: Marks the beginning of speech generation
    - speech_end: Marks the end of speech generation
    - speech_diffusion: Placeholder for diffusion tokens
    
    Example:
        >>> tokenizer = KugelAudioTextTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        >>> tokens = tokenizer.encode("Hello world")
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        self._add_speech_special_tokens()
    
    def _add_speech_special_tokens(self):
        """Add KugelAudio-specific special tokens for speech."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens for compatibility)
                "<|vision_end|>",    # Speech end
                "<|vision_pad|>",    # Speech diffusion pad
            ]
        }
        self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")
        self._eos_id = self.eos_token_id
        self._pad_id = self.convert_tokens_to_ids("<|image_pad|>")
    
    @property
    def eos_id(self) -> int:
        """End of sequence token ID."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Speech start token ID."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Speech end token ID."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Speech diffusion placeholder token ID."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Padding token ID (returns -100 for loss masking)."""
        return self._pad_id
