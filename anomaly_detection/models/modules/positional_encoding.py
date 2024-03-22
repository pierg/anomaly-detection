"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from calendar import c
from loguru import logger
import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding introduces a unique position-specific signal to token embeddings
    to retain sequence order information. It uses a mix of sine and cosine functions with different
    frequencies to generate a distinct and deterministic positional encoding for each sequence position.
    
    Unlike embedding-based positional encodings, sinusoidal encodings are not learned from data.
    Instead, they are computed based on their position in the sequence and the embedding dimension,
    ensuring that each position maps to a unique, fixed vector. This allows models to interpolate
    and understand sequence positions never seen during training, supporting variable sequence lengths
    and generalization.
    
    Parameters:
    - d_model (int): The dimensionality of the token embeddings and positional encodings.
    - max_len (int): The maximum sequence length for which positional encodings will be precomputed.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        logger.info(f"Initializing sinusoidal positional encoding with d_model={d_model} and max_len={max_len}")
        
        # Initialize positional encoding matrix with zeros.
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension (at position 0) for easy addition to token embeddings.
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Register as a buffer to avoid being considered a model parameter.

    def forward(self, x):
        """
        Adds sinusoidal positional encodings to the token embeddings.
        
        Parameters:
        - x (Tensor): The token embeddings with shape [batch_size, sequence_length, d_model].
        
        Returns:
        - Tensor: The token embeddings with added positional encodings, same shape as input.
        """
        # Adjust the slicing of the positional encoding to match the sequence length of x.
        # No need to transpose x as it's already in the desired shape [batch_size, sequence_length, d_model].
        # Use broadcasting to add the positional encoding across the batch dimension.
        x = x + self.pe[:, :x.size(1)].expand_as(x)
        return x


class EmbeddingPositionalEncoding(nn.Module):
    """
    Embedding Positional Encoding uses an embedding layer to learn positional encodings from data,
    as opposed to using predefined functions. Each position in a sequence is mapped to a learnable
    vector, allowing the model to adapt positional encodings to the specific requirements of the task.
    
    This approach treats positions as discrete entities, similar to word tokens, and leverages the
    model's learning capacity to optimize how position information is represented. It's particularly
    effective when the model needs to learn complex or non-linear positional relationships within
    sequences.
    
    Parameters:
    - d_model (int): The dimensionality of the token embeddings and positional encodings.
    - max_len (int): The maximum sequence length for which positional embeddings will be learned.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)  # Learnable embedding for positions.

    def forward(self, x):
        """
        Adds learned positional encodings to the token embeddings.
        
        Parameters:
        - x (Tensor): The token embeddings with shape [batch_size, sequence_length, d_model].
        
        Returns:
        - Tensor: The token embeddings with added positional encodings, same shape as input.
        """

        device = x.device  # Determine the device (CPU/GPU) where the input tensor resides to ensure compatibility.
        B, T, _ = x.shape  # Extract the batch size (B), sequence length (T), and d_model from the input tensor's shape.

        # Generate position indices for each element in the sequence.
        position_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # Shape [B, T]

        # Ensure that the learned positional encodings can accept position_indices of shape [B, T]
        # and understand you'll need to have your pos_embedding layer structured to handle this input shape,
        # possibly with some form of embedding lookup that supports batched indices.
        pos_encodings = self.pos_embedding(position_indices)  # Expected shape [B, T, d_model]

        # Add the positional encodings to the input embeddings.
        # Assuming pos_encodings is correctly shaped as [B, T, d_model], it can be directly added to x.
        x = x + pos_encodings
        return x


