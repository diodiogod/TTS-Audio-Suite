import torch


class DualCodebookEmbedding(torch.nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 input_size: int,
                 ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, input_size // 2)
        
    def forward(self, token: torch.Tensor):
        """
        Args:
            token (torch.Tensor): shape (b, t, 2)
        Returns:
            xs: shape (b, t, c)
        """
        embed1 = self.embedding(token[..., 0])
        embed2 = self.embedding(token[..., 1])
        return torch.cat([embed1, embed2], dim=-1)

