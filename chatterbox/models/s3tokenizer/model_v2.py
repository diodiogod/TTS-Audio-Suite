import torch
import torch.nn as nn
from typing import Tuple


class ModelConfig:
    """Configuration class for S3TokenizerV2 model."""
    
    def __init__(self, n_mels: int = 128):
        self.n_mels = n_mels


class S3TokenizerV2(nn.Module):
    """
    Base Speech Tokenizer V2 model.
    This is a simplified implementation that can be extended.
    """
    
    def __init__(self, name: str = "speech_tokenizer_v2_25hz"):
        super().__init__()
        self.name = name
        self.device = torch.device("cpu")
        
        # These would typically be loaded from pretrained model files
        # For now, we create placeholder networks
        self.encoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Quantization layers - simplified version
        self.quantize_conv = nn.Conv1d(1024, 512, 1)
        self.quantize_layer = nn.Linear(512, 6561)  # SPEECH_VOCAB_SIZE
        
    def to(self, device):
        """Move model to device and update internal device tracking."""
        super().to(device)
        self.device = device
        return self
        
    @torch.no_grad()
    def quantize(self, mels: torch.Tensor, mel_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize mel spectrograms to discrete speech tokens.
        
        Args:
            mels: Tensor of shape [batch_size, n_mels, time_steps]
            mel_lens: Tensor of shape [batch_size] containing sequence lengths
            
        Returns:
            speech_tokens: Tensor of shape [batch_size, time_steps//4] containing token indices
            speech_token_lens: Tensor of shape [batch_size] containing token sequence lengths
        """
        batch_size, n_mels, time_steps = mels.shape
        
        # Move to same device as model
        mels = mels.to(self.device)
        mel_lens = mel_lens.to(self.device)
        
        # Encode the mel spectrograms
        encoded = self.encoder(mels)  # [batch_size, 1024, time_steps]
        
        # Quantize
        quantized = self.quantize_conv(encoded)  # [batch_size, 512, time_steps]
        
        # Downsample by factor of 4 (25 tokens/sec from 100 frames/sec)
        quantized = quantized[:, :, ::4]  # [batch_size, 512, time_steps//4]
        
        # Convert to discrete tokens
        quantized = quantized.transpose(1, 2)  # [batch_size, time_steps//4, 512]
        speech_tokens = self.quantize_layer(quantized)  # [batch_size, time_steps//4, 6561]
        speech_tokens = torch.argmax(speech_tokens, dim=-1)  # [batch_size, time_steps//4]
        
        # Compute token lengths (downsampled by factor of 4)
        speech_token_lens = (mel_lens.float() / 4).ceil().long()
        
        return speech_tokens, speech_token_lens
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict, handling missing keys gracefully."""
        # Handle ignore_state_dict_missing attribute from S3Tokenizer
        if hasattr(self, 'ignore_state_dict_missing'):
            for key in self.ignore_state_dict_missing:
                if key in state_dict:
                    del state_dict[key]
        
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if not strict:
                print(f"Warning: Some keys were not loaded: {e}")
                return super().load_state_dict(state_dict, strict=False)
            else:
                raise e