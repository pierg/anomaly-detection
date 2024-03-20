"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT_v3(nn.Module):
    """
    GPT_v3 model with positional embeddings and a sliding window generation approach. This model
    captures the sequential nature of language by utilizing both token and positional embeddings,
    and generates text by considering only the most recent tokens within its maximum sequence length.
    """
    
    def __init__(self, vocab_size: int, n_embd: int, block_size: int) -> None:
        """
        Initializes the model with token and positional embeddings, and a linear transformation layer.

        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - n_embd (int): Size of each embedding vector.
        - block_size (int): Maximum length of the input sequences.
        """
        super(GPT_v3, self).__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining token and positional embeddings to produce logits.

        Parameters:
        - indices (torch.Tensor): Input token indices (batch_size, sequence_length).

        Returns:
        - torch.Tensor: Logits (batch_size, sequence_length, vocab_size).
        """
        B, T = indices.shape
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        position_indices = torch.arange(T, device=device).expand(B, -1)  # Position indices for the sequence
        tok_emb = self.token_embedding_table(indices)  # Token embeddings
        pos_emb = self.position_embedding_table(position_indices)  # Positional embeddings
        x = tok_emb + pos_emb  # Combine embeddings
        logits = self.lm_head(x)  # Linear transformation to logits
        return logits

    def generate(self, start_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates text by considering the most recent part of the sequence that fits within
        the model's processing capacity (block_size).

        Parameters:
        - start_indices (torch.Tensor): Starting sequence of token indices.
        - max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
        - torch.Tensor: Extended sequence of token indices after generation.
        """
        # Compute block_size based on the number of position embeddings available
        block_size = self.position_embedding_table.weight.size(0)

        indices = start_indices
        for _ in range(max_new_tokens):
            # Use the most recent 'block_size' tokens to generate the next token
            current_indices = indices[:, -block_size:]
            logits = self(current_indices)  # Obtain logits for the current sequence "window"
            logits = logits[:, -1, :]  # Focus on the logits for the last token in the sequence
            probabilities = F.softmax(logits, dim=-1)  # Compute probabilities
            next_index = torch.multinomial(probabilities, num_samples=1)  # Sample the next token
            indices = torch.cat((indices, next_index), dim=1)  # Append the new token to the sequence

        return indices
