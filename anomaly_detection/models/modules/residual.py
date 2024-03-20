"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch.nn as nn
from models.modules.multi_head import MultiHeadAttention
from models.modules.computation import FeedForward

class Block(nn.Module):
    """
    A Transformer block that consists of a multi-head self-attention mechanism ('communication')
    followed by a position-wise feedforward network ('computation'), with each sub-layer being
    preceded by layer normalization and followed by a residual connection.
    """
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        """
        Initializes the Transformer block.

        Parameters:
        - n_embd (int): The embedding dimension.
        - n_head (int): The number of attention heads.
        - block_size (int): The maximum length of the input sequences.
        - dropout (float): Dropout rate for regularization within the MultiHeadAttention and FeedForward layers.
        """
        super(Block, self).__init__()
        self.head_size = n_embd // n_head  # Calculate the size of each head dynamically
        self.sa = MultiHeadAttention(num_heads=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)  # Normalizes features across the embedding dimension.
        self.ln2 = nn.LayerNorm(n_embd)  # Same as above, applied after the feedforward network.

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd).

        Returns:
        - Tensor: Output tensor of the same shape as input.
        """
        # LayerNorm before self-attention helps stabilize the inputs to the attention mechanism
        # by ensuring that they have a mean of 0 and a standard deviation of 1. This normalization
        # is crucial for preventing the self-attention outputs from having excessively large or
        # small scales, which can lead to unstable gradients and hinder the learning process.
        x_ln1 = self.ln1(x)
        x_sa = self.sa(x_ln1)
        x = x + x_sa
        
        # Applying LayerNorm before the feedforward network similarly stabilizes the inputs
        # to this network. By doing so, it ensures a consistent distribution of inputs across
        # layers, facilitating smoother gradient flow and more stable training, especially in
        # deep models. The subsequent residual connection allows for the integration of the
        # original input and the transformed output, promoting effective information flow.
        x_ln2 = self.ln2(x)
        x_ffwd = self.ffwd(x_ln2)
        x = x + x_ffwd
        return x