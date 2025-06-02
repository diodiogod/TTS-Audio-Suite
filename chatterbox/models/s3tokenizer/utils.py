import torch
import torch.nn.functional as F


def padding(mels):
    """
    Pad a list of mel spectrograms to the same length.
    
    Args:
        mels: List of mel spectrograms, each of shape [n_mels, time_steps]
        
    Returns:
        padded_mels: Tensor of shape [batch_size, n_mels, max_time_steps]
        mel_lens: Tensor of shape [batch_size] containing the original lengths
    """
    if not mels:
        return torch.empty(0), torch.empty(0)
    
    # Get the lengths of each mel spectrogram
    mel_lens = torch.tensor([mel.shape[-1] for mel in mels])
    
    # Find the maximum length
    max_len = mel_lens.max().item()
    
    # Pad all mel spectrograms to the same length
    padded_mels = []
    for mel in mels:
        if mel.dim() == 2:  # [n_mels, time_steps]
            pad_amount = max_len - mel.shape[-1]
            if pad_amount > 0:
                padded_mel = F.pad(mel, (0, pad_amount), mode='constant', value=0)
            else:
                padded_mel = mel
        else:
            raise ValueError(f"Expected mel spectrogram to have 2 dimensions, got {mel.dim()}")
        
        padded_mels.append(padded_mel)
    
    # Stack into a batch
    padded_mels = torch.stack(padded_mels, dim=0)
    
    return padded_mels, mel_lens