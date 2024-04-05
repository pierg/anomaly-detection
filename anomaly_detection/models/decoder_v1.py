"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT_v1(nn.Module):
    """
    A simplistic implementation of a Bigram Language Model that uses an embedding layer to map
    vocabulary indices to dense vector representations. This model is the first step in building
    towards more complex Transformer-based models.
    """

    def __init__(self, vocab_size: int) -> None:
        """
        Initializes the GPT model with a single embedding layer.

        Parameters:
        - vocab_size (int): The size of the vocabulary, determining the number of unique tokens
                            that can be represented.

        The embedding layer now correctly represents a mapping from token indices to their embedding
        vectors, with dimensions `(vocab_size, embedding_dim)`, where `embedding_dim` is chosen to
        be equal to `vocab_size` for simplicity in this initial model version.
        """
        super(GPT_v1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Parameters:
        - indices (torch.Tensor): A tensor of token indices with shape `(batch_size, sequence_length)`.

        Returns:
        - torch.Tensor: The logits (unnormalized predictions) with shape `(batch_size, sequence_length, vocab_size)`.

        The embedding layer transforms the input indices into dense vectors, serving as a simple form
        of 'understanding' of the input sequence.
        """
        logits = self.embedding(indices)
        return logits

    def generate(
        self, start_indices: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """
        Generates new tokens based on the given starting sequence.

        Parameters:
        - start_indices (torch.Tensor): The starting sequence of token indices.
        - max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
        - torch.Tensor: The extended sequence of token indices after generation.

        This method iteratively predicts the next token based on the last token's embedding,
        showcasing a very basic form of text generation.
        """
        indices = start_indices
        for _ in range(max_new_tokens):
            logits = self(indices)  # Obtain logits for the current sequence
            # Focus on the logits for the last token in the sequence
            logits = logits[:, -1, :]
            # Compute softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(
                probabilities, num_samples=1
            )  # Sample the next token
            # Append the new token to the sequence
            indices = torch.cat((indices, next_index), dim=1)

        return indices
