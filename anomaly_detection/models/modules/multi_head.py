"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
from models.modules.head import Head


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention, running several attention mechanisms (heads) in parallel,
    and then combining their outputs. This allows the model to capture information from different
    representation subspaces. After concatenating these outputs, a projection is applied to return
    to the original embedding size, facilitating the use of residual connections in a Transformer model.

    This implementation automatically computes the size of each head based on the embedding dimension
    (n_embd) and the number of heads (num_heads), ensuring that the total dimensionality of the output
    from all heads combined matches the input embedding dimensionality. This constraint is necessary
    to allow the seamless integration of the Multi-Head Attention output with subsequent layers in the model.
    """
    
    def __init__(self, num_heads: int, n_embd: int, block_size: int, dropout: float = 0.1):
        """
        Initializes the Multi-Head Attention module.

        Parameters:
        - num_heads (int): Number of attention heads.
        - n_embd (int): Total dimension of the input embeddings.
        - block_size (int): Maximum length of the input sequence.
        - dropout (float): Dropout rate for regularization.
        """
        super(MultiHeadAttention, self).__init__()
        # Ensure the embedding dimension can be evenly divided by the number of heads
        assert n_embd % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        self.head_size = n_embd // num_heads  # Dimensionality of each head's output
        
        self.heads = nn.ModuleList([
            Head(head_size=self.head_size, n_embd=n_embd, block_size=block_size, dropout=dropout)
            for _ in range(num_heads)
        ])

        # Projecting back to n_embd dimensions
        self.proj = nn.Linear(self.head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Multi-Head Attention layer.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, n_embd).

        Returns:
        - torch.Tensor: Output tensor with the same shape as the input, after applying
          multi-head attention and projecting back to the original dimension.
        """
        # Concatenate the outputs of all heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # Project back to n_embd dimensions and apply dropout
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

