"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from anomaly_detection.models.modules.head import Head


class GPT_v4(nn.Module):
    """
    GPT_v4 integrates self-attention into the GPT model architecture, enhancing its ability
    to understand and generate language by allowing it to focus on relevant parts of the input
    sequence. This version introduces a self-attention head (sa_head) to process the input
    embeddings, incorporating both positional information and learned attention patterns.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        block_size: int,
        head_size: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes the GPT_v4 model with token and positional embeddings, a self-attention head,
        and a linear layer for the language model head.

        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - n_embd (int): Size of each embedding vector.
        - block_size (int): Maximum length of the input sequences.
        - head_size (int): Dimensionality of the attention head.
        - dropout (float): Dropout rate for the attention mechanism.
        """
        super(GPT_v4, self).__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Initialize the self-attention head with the correct parameters
        # Since we only have one head, we can set the head_size to be equal to
        # n_embd
        assert (
            head_size == n_embd
        ), "Head size must be equal to the embedding size for a single head self-attention"
        self.sa_head = Head(
            head_size=head_size, n_embd=n_embd, block_size=block_size, dropout=dropout
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT_v4 model, applying token and positional embeddings followed by
        self-attention and a final linear transformation to produce logits.

        Parameters:
        - indices (torch.Tensor): Input token indices (batch_size, sequence_length).

        Returns:
        - torch.Tensor: Output logits (batch_size, sequence_length, vocab_size).
        """
        B, T = indices.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        position_indices = torch.arange(T, device=device).expand(
            B, -1
        )  # Position indices for the sequence
        tok_emb = self.token_embedding_table(indices)  # Token embeddings
        pos_emb = self.position_embedding_table(
            position_indices
        )  # Positional embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        x = self.sa_head(x)  # Apply self-attention head
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
