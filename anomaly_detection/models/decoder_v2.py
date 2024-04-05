"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT_v2(nn.Module):
    """
    An evolved version of the GPT model that introduces a linear layer (language model head) on top of the embeddings.
    This architecture allows for a more sophisticated mapping from the token embeddings to the vocabulary space,
    facilitating the learning of richer representations for sequence prediction and generation.

    Compared to GPT_v1, this model adds a linear transformation layer, enhancing its ability to capture the nuances
    of the language by providing a flexible mechanism for generating logits, rather than relying solely on embeddings.
    """

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        """
        Initializes the GPT_v2 model with an embedding layer and a linear layer for the language model head.

        Parameters:
        - vocab_size (int): The size of the vocabulary, determining the number of unique tokens.
        - n_embd (int): The size of each embedding vector.

        The model architecture now includes a token embedding table for converting token indices to embeddings
        and a linear layer (language model head) that maps embeddings to logits across the vocabulary.
        """
        super(GPT_v2, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Parameters:
        - indices (torch.Tensor): A tensor of token indices with shape `(batch_size, sequence_length)`.

        Returns:
        - torch.Tensor: The logits with shape `(batch_size, sequence_length, vocab_size)`.

        This method computes token embeddings and applies a linear transformation to produce logits,
        enabling a richer understanding and generation of sequences.
        """
        tok_emb = self.token_embedding_table(indices)  # Convert indices to embeddings
        # Apply linear layer to embeddings to get logits
        logits = self.lm_head(tok_emb)
        return logits

    def generate(
        self, start_indices: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        """
        Generates new tokens based on the given starting sequence, using the model's current understanding
        and prediction capabilities.

        Parameters:
        - start_indices (torch.Tensor): The starting sequence of token indices.
        - max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
        - torch.Tensor: The extended sequence of token indices after generation.

        This method iteratively generates new tokens by predicting the next token based on the current sequence,
        showcasing the model's ability to extend sequences in a contextually relevant manner.
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
