"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly_detection.models.modules.residual import Block


class GPT_v7(nn.Module):
    """
    GPT_v7 model extends the GPT architecture with multiple Transformer blocks, enhancing its
    ability to process complex sequences. This architecture leverages deep self-attention mechanisms
    and position-wise feedforward networks, encapsulated within residual blocks, for improved
    representation learning. The model concludes with a final layer normalization and a linear
    projection to vocabulary space, enabling effective sequence generation and analysis.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_layers: int = 3,
    ) -> None:
        """
        Initializes the GPT_v7 model components.

        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - n_embd (int): Dimensionality of the embedding space.
        - block_size (int): Maximum length of the input sequences.
        - dropout (float): Dropout rate applied within blocks for regularization.
        - num_heads (int): Number of heads in the multi-head attention mechanisms.
        - num_layers (int): Number of Transformer blocks to stack.
        """
        super(GPT_v7, self).__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Initializing a sequence of Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=num_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization for stabilizing the output of the last
        # block
        self.ln_f = nn.LayerNorm(n_embd)

        # Linear projection to the vocabulary size
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the GPT_v7 model.

        Parameters:
        - indices (torch.Tensor): Input token indices with shape (batch_size, sequence_length).

        Returns:
        - torch.Tensor: Output logits over the vocabulary for each position in the sequence.
        """
        device = indices.device
        B, T = indices.shape
        position_indices = torch.arange(T, device=device).expand(
            B, -1
        )  # Generate position indices for each sequence element

        # Embedding tokens and positions, and combining them
        tok_emb = self.token_embedding_table(indices)
        pos_emb = self.position_embedding_table(position_indices)
        x = tok_emb + pos_emb  # Summing token and position embeddings

        # Passing the combined embeddings through the sequence of Transformer
        # blocks
        x = self.blocks(x)

        # Applying final layer normalization
        x = self.ln_f(x)

        # Projecting to vocabulary space to produce logits
        logits = self.lm_head(x)

        return logits

    def generate(
        self, start_indices: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """
        Generates text by considering the most recent part of the sequence that fits within
        the model's processing capacity (block_size).

        Parameters:
        - start_indices (torch.Tensor): Starting sequence of token indices.
        - max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
        - torch.Tensor: Extended sequence of token indices after generation.
        """
        # Compute block_size based on the number of position embeddings
        # available
        block_size = self.position_embedding_table.weight.size(0)

        indices = start_indices
        for _ in range(max_new_tokens):
            # Use the most recent 'block_size' tokens to generate the next
            # token
            current_indices = indices[:, -block_size:]
            # Obtain logits for the current sequence "window"
            logits = self(current_indices)
            # Focus on the logits for the last token in the sequence
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)  # Compute probabilities
            next_index = torch.multinomial(
                probabilities, num_samples=1
            )  # Sample the next token
            # Append the new token to the sequence
            indices = torch.cat((indices, next_index), dim=1)

        return indices
