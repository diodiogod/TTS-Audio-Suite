import torch
import torch.nn as nn


class PartiallyFrozenEmbedding(nn.Module):
    """Split an existing `nn.Embedding` module that splits the embedding into:

    - A frozen embedding for indices [0..freeze_until_idx].
    - A trainable embedding for indices [freeze_until_idx+1..vocab_size-1].

    This should work with both Zero-2 and Zero-3 seamlessly
    """

    def __init__(self, original_embedding: nn.Embedding, freeze_until_idx: int):
        """
        :param original_embedding: An instance of nn.Embedding (the original embedding layer).
        :param freeze_until_idx: The index up to which the embedding is frozen (excluding). The freeze_until_idx is not frozen.
        """
        super().__init__()
        self.freeze_until_idx = freeze_until_idx
        self.original_vocab_size = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim

        # Split the original embedding into frozen and trainable parts
        self.embedding_frozen = nn.Embedding(
            freeze_until_idx,
            self.embedding_dim,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device,
        )
        self.embedding_trainable = nn.Embedding(
            self.original_vocab_size - freeze_until_idx,
            self.embedding_dim,
            dtype=original_embedding.weight.dtype,
            device=original_embedding.weight.device,
        )

        # Copy weights from the original embedding into the frozen and trainable parts
        with torch.no_grad():
            self.embedding_frozen.weight.copy_(original_embedding.weight[:freeze_until_idx])
            self.embedding_trainable.weight.copy_(original_embedding.weight[freeze_until_idx:])

        # Freeze the frozen embedding
        self.embedding_frozen.weight.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the split embedding wrapper.
        :param input_ids: Tensor of shape [batch_size, seq_len] with indices in [0..original_vocab_size-1].
        """
        # Masks to separate frozen and trainable indices
        # (bsz, seq_len)
        mask_frozen = input_ids < self.freeze_until_idx
        mask_trainable = ~mask_frozen

        # Output tensor for embedding results
        batch_size, seq_len = input_ids.shape
        embeddings = torch.zeros(
            batch_size,
            seq_len,
            self.embedding_dim,
            device=input_ids.device,
            dtype=self.embedding_frozen.weight.dtype,
        )

        # Handle frozen embedding
        if mask_frozen.any():
            frozen_ids = input_ids[mask_frozen]
            frozen_emb = self.embedding_frozen(frozen_ids)
            embeddings[mask_frozen] = frozen_emb

        # Handle trainable embedding
        if mask_trainable.any():
            # Adjust trainable IDs to the local index space of the trainable embedding
            trainable_ids = input_ids[mask_trainable] - (self.freeze_until_idx)
            trainable_emb = self.embedding_trainable(trainable_ids)
            embeddings[mask_trainable] = trainable_emb

        return embeddings

    def to_unsplit(self) -> nn.Embedding:
        unsplit_embedding = nn.Embedding(
            self.original_vocab_size,
            self.embedding_dim,
            dtype=self.embedding_frozen.weight.dtype,
            device=self.embedding_frozen.weight.device,
        )

        with torch.no_grad():
            unsplit_embedding.weight[: self.freeze_until_idx].copy_(self.embedding_frozen.weight)
            unsplit_embedding.weight[self.freeze_until_idx :].copy_(self.embedding_trainable.weight)

        return unsplit_embedding


class PartiallyFrozenLinear(nn.Module):
    """A wrapper around nn.Linear to partially freeze part of the weight matrix."""

    def __init__(self, original_linear: nn.Linear, freeze_until_idx: int):
        """
        :param original_linear: The original nn.Linear layer.
        :param freeze_until_idx: The index up to which the rows of the weight matrix are frozen.
        """
        super().__init__()
        assert original_linear.bias is None, "Currently only support linear module without bias"

        self.freeze_until_idx = freeze_until_idx
        self.input_dim = original_linear.in_features
        self.output_dim = original_linear.out_features

        # Create frozen and trainable linear layers
        self.linear_frozen = nn.Linear(
            self.input_dim,
            freeze_until_idx,
            bias=False,
            dtype=original_linear.weight.dtype,
            device=original_linear.weight.device,
        )
        self.linear_trainable = nn.Linear(
            self.input_dim,
            self.output_dim - freeze_until_idx,
            bias=False,
            dtype=original_linear.weight.dtype,
            device=original_linear.weight.device,
        )

        # Copy weights from the original linear layer
        with torch.no_grad():
            self.linear_frozen.weight.copy_(original_linear.weight[:freeze_until_idx])
            self.linear_trainable.weight.copy_(original_linear.weight[freeze_until_idx:])

        # Freeze the frozen linear layer
        self.linear_frozen.weight.requires_grad = False

    def forward(self, input_tensor):
        # input_tensor: (bsz, seq_len, hidden_state_dim)
        frozen_output = self.linear_frozen(input_tensor)
        trainable_output = self.linear_trainable(input_tensor)
        return torch.cat((frozen_output, trainable_output), dim=-1)

    def to_unsplit(self) -> nn.Linear:
        unsplit_linear = nn.Linear(
            self.input_dim,
            self.output_dim,
            bias=False,
            dtype=self.linear_frozen.weight.dtype,
            device=self.linear_frozen.weight.device,
        )

        # Copy weights from the frozen and trainable layers into the unsplit linear layer
        with torch.no_grad():
            unsplit_linear.weight[: self.freeze_until_idx].copy_(self.linear_frozen.weight)
            unsplit_linear.weight[self.freeze_until_idx :].copy_(self.linear_trainable.weight)

        return unsplit_linear
