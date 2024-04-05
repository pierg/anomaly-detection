"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
from loguru import logger

from anomaly_detection.models.modules.positional_encoding import (
    EmbeddingPositionalEncoding, SinusoidalPositionalEncoding)
from anomaly_detection.models.modules.residual import Block


class TransformerDL(nn.Module):
    """
    TransformerDL model integrates a series of Transformer blocks with positional encoding
    to process sequences. This architecture is particularly useful for tasks requiring an
    understanding of both the content and the order within sequences, like time series analysis
    or sequence classification.

    The model comprises an input projection layer, positional encoding, Transformer blocks,
    and an output layer for final predictions, accommodating the requirement to grasp the sequential
    context and dependencies across elements in the input sequence.
    """

    def __init__(
        self,
        input_size,
        d_model,
        block_size,
        n_heads,
        n_blocks,
        output_size,
        pos_embedding="sinusoidal",
        dropout=0.1,
    ):
        """
        Initializes the TransformerDL model with the specified architecture components.

        Parameters:
        - input_size (int): The size of each input feature. For time series data, this is expected to be 1.
        - d_model (int): The dimensionality for the internal embeddings of the model.
        - block_size (int): The maximum length of input sequences to be considered for positional encoding.
        - n_heads (int): The number of attention heads within each Transformer block.
        - n_blocks (int): The total number of Transformer blocks to be included in the model.
        - output_size (int): The dimensionality of the output layer, suitable for the task at hand.
        - pos_embedding (str, optional): The type of positional encoding to use ('sinusoidal' or others). Defaults to 'sinusoidal'.
        - dropout (float, optional): The dropout rate to apply within the model. Defaults to 0.1.
        """
        super(TransformerDL, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding selection based on the pos_embedding argument
        if pos_embedding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, block_size)
        else:
            # Assumes alternative is the embedding-based positional encoding
            self.pos_encoder = EmbeddingPositionalEncoding(d_model, block_size)

        # Initializing the Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embd=d_model, n_head=n_heads, block_size=block_size, dropout=0.1
                )
                for _ in range(n_blocks)
            ]
        )

        # Output layer for the model predictions
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the TransformerDL model.

        Parameters:
        - x (Tensor): The input tensor with shape [batch_size, sequence_length, input_size].

        Returns:
        - Tensor: The output tensor with shape [batch_size, sequence_length, output_size].
        """
        logger.debug(f"x:\t{x.shape}\t{x.dtype}")

        # Project input to match d_model dimensions
        x = self.input_projection(x)

        logger.debug(f"x:\t{x.shape}\t{x.dtype}")

        # Apply positional encoding
        x = self.pos_encoder(x)

        logger.debug(f"x:\t{x.shape}\t{x.dtype}")

        # Pass x through each Transformer block
        for block in self.blocks:
            x = block(x)

        # Select the output of the last token
        x = x[:, -1, :]  # Shape: [batch_size, d_model]
        logger.debug(f"x:\t{x.shape}\t{x.dtype}")

        # Final prediction
        x = self.output_layer(x)  # Shape: [batch_size, output_size]

        logger.debug(f"x:\t{x.shape}\t{x.dtype}")

        return x
