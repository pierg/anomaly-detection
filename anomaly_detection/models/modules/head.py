"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    Implements a single head of self-attention mechanism, which is a fundamental component of Transformer architectures.
    This mechanism allows the model to weigh the importance of different parts of the input sequence differently when
    predicting an output.
    
    The self-attention mechanism uses queries, keys, and values derived from the input embeddings, with the attention
    scores dictating the focus on different input parts. A mask is applied to ensure the model does not attend to future
    tokens, aligning with the requirements of a decoder to enforce causality.
    """
    
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float = 0.1) -> None:
        """
        Initializes the self-attention head components, including linear transformations for queries, keys, and values,
        and a dropout layer for the attention scores.

        Parameters:
        - head_size (int): Dimensionality of each attention head.
        - n_embd (int): Size of the input embedding vectors.
        - block_size (int): Maximum length of the input sequence, used for creating the mask.
        - dropout (float): Dropout rate applied to the attention scores to prevent overfitting.
        """
        super(Head, self).__init__()
        self.head_size = head_size

        self.key = nn.Linear(n_embd, head_size, bias=False)  # Key transformation
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Query transformation
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Value transformation
        self.dropout = nn.Dropout(dropout)

        # Lower triangular matrix for masking future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention head, computing weighted sum of value vectors based on attention scores.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (B, T, C), where B is the batch size, T is the sequence length,
                            and C is the embedding size.

        Returns:
        - torch.Tensor: Output tensor after applying self-attention, with shape (B, T, head_size).
        """
        B, T, C = x.shape  # Extract dimensions for clarity

        # Transform input to query, key, and value vectors
        # Shapes: (B, T, C) -> (B, T, head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores using query and key
        # Shape transformation: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1)

        # Mask future tokens by setting attention scores to -inf where appropriate
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Scale attention scores by the square root of the head size
        # This scaling helps stabilize gradients by preventing the softmax input from becoming too large,
        # which can cause the gradients to be too small and slow down the learning. It effectively normalizes
        # the dot product's magnitude to a reasonable range.
        wei = wei / (self.head_size ** 0.5)

        # Normalize attention scores to probabilities
        # Shape: (B, T, T), softmax over the last dimension
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # Apply dropout, shape remains (B, T, T)

        # Compute the weighted sum of value vectors based on attention scores
        # Shape transformation: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v

        return out
