"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly_detection.models.modules.multi_head import MultiHeadAttention


class GPT_v5(nn.Module):
    """
    GPT_v5 enhances the GPT architecture by incorporating a multi-head attention mechanism,
    allowing the model to attend to different parts of the sequence simultaneously. This
    approach enables capturing a more comprehensive understanding of the sequence's context
    and relationships. The model also includes token and positional embeddings, along with
    a linear layer for generating logits corresponding to the vocabulary predictions.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        head_size: int,
        dropout: float = 0.1,
        num_heads: int = 4,
    ) -> None:
        """
        Initializes the GPT_v5 model.

        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - n_embd (int): The size of the embedding vectors.
        - block_size (int): The maximum length of the input sequences.
        - head_size (int): The size of each attention head (not used here since head_size is inferred inside MultiHeadAttention).
        - dropout (float): The dropout rate.
        - num_heads (int): The number of attention heads.
        """
        super(GPT_v5, self).__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(
            num_heads=num_heads, n_embd=n_embd, block_size=block_size, dropout=dropout
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT_v5 model.

        Parameters:
        - indices (torch.Tensor): The input token indices.

        Returns:
        - torch.Tensor: The logits corresponding to the vocabulary predictions.
        """
        device = indices.device  # Optimized device handling
        B, T = indices.shape
        position_indices = torch.arange(T, device=device).expand(
            B, -1
        )  # Position indices for the sequence
        tok_emb = self.token_embedding_table(indices)  # Token embeddings
        pos_emb = self.position_embedding_table(
            position_indices
        )  # Positional embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.sa_heads(x)  # Apply multi-head self-attention
        logits = self.lm_head(x)  # Apply linear layer to produce logits
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
