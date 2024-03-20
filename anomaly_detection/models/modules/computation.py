"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Defines a simple feedforward neural network module used within the Transformer architecture,
    featuring an expanded hidden layer to increase the model's capacity for learning complex functions.
    The expansion factor (i.e. 4) is a empirical choice in the original Transformer model in the paper "Attention is All You Need".
    This design choice follows the principle of increasing the dimensionality of the intermediate layer
    to provide a richer representation space before compressing the outputs back to the original dimension.
    """
    
    def __init__(self, n_embd: int, dropout: float = 0.1):
        """
        Initializes the feedforward network.

        Parameters:
        - n_embd (int): The size of the input and output dimensions.
        - dropout (float): The dropout rate for regularization, helping prevent overfitting.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expansion layer: Increases capacity and allows for capturing more complex patterns.
            nn.ReLU(),                     # Introduces non-linearity, enabling the network to learn non-linear functions.
            nn.Linear(4 * n_embd, n_embd), # Compression layer: Reduces dimensionality back, focusing on the most relevant features.
            nn.Dropout(dropout),           # Applies dropout for regularization, improving generalization.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the feedforward network.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after processing by the feedforward network.
        """
        return self.net(x)
