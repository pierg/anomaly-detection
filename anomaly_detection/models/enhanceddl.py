"""
Author: Piergiuseppe Mallozzi
Date: 2024

This file contains the EnhancedDL model, which is a combination of LSTM and Transformer blocks.
"""

import torch
import torch.nn as nn
from loguru import logger

from anomaly_detection.models.modules.residual import Block


class EnhancedDL(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the EnhancedDL model with both LSTM and Transformer blocks.

        Parameters:
        - input_size (int): The size of input features.
        - hidden_size (int): The size of hidden units in LSTM.
        - num_layers (int): Number of LSTM layers.
        - output_size (int): The size of the output layer.
        - n_heads (int): Number of heads in the MultiHeadAttention module.
        - dropout (float, optional): Dropout rate for regularization. Defaults to 0.1.
        """
        super(EnhancedDL, self).__init__()

        # Logging the model's architecture initialization details
        logger.info(
            f"Initializing EnhancedDL with input size: {input_size}, hidden size: {hidden_size}, "
            f"num layers: {num_layers}, output size: {output_size}, n_heads: {n_heads}, dropout: {dropout}"
        )

        # LSTM layer for initial sequential processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Transformer block for capturing complex dependencies
        # Input and output sizes of the transformer block are set to
        # hidden_size, as it processes LSTM's outputs
        self.transformer_block = Block(
            n_embd=hidden_size, n_head=n_heads, block_size=hidden_size, dropout=dropout
        )

        # Final output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EnhancedDL model.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after processing by the LSTM, Transformer block, and the linear layer.
        """

        # Log the size and type of the input tensor
        logger.debug(f"Input tensor size: {x.shape}, type: {x.dtype}")

        # Passing input through LSTM layer
        x, _ = self.lstm(x)
        logger.debug(f"LSTM output size: {x.shape}, type: {x.dtype}")

        # Passing LSTM output through Transformer block
        x = self.transformer_block(x)
        logger.debug(f"Transformer block output size: {x.shape}, type: {x.dtype}")

        # Selecting the output from the last timestep and passing it through
        # the linear layer
        x = self.fc(x[:, -1, :])
        logger.debug(f"Final output size: {x.shape}, type: {x.dtype}")

        return x
