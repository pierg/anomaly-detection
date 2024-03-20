"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
from typing import Tuple, Dict
from data.base_data import DataLoader, Tokenizer, BatchGenerator

class TextDataLoader(DataLoader):
    """Data loader for text files."""

    def load_data(self, file_path: str) -> str:
        """Reads and returns content of a text file."""
        with open(file_path, 'r') as file:
            return file.read()

class CharacterTokenizer(Tokenizer):
    """Character-level tokenizer for text."""

    def __init__(self, text: str):
        # Create mappings from characters to indices and vice versa
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def tokenize(self, text: str) -> torch.Tensor:
        """Convert text to a tensor of indices."""
        return torch.tensor([self.stoi[char] for char in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """Convert tensor of indices back to text."""
        return ''.join(self.itos[int(token)] for token in tokens)

    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.chars)

class TextBatchGenerator(BatchGenerator):
    """Batch generator for tokenized text data."""

    def __init__(self, data_splits: Dict[str, torch.Tensor], block_size: int):
        """
        Initialize with multiple data splits and a block size.
        'data_splits' is a dictionary with keys as split names and values as data tensors.
        'block_size' is size of each block of text to be used as input, serving as a prompt/context.
        """
        self.data_splits = data_splits
        self.block_size = block_size

    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates a batch of data for the given split. """

        # Choose the appropriate dataset based on the split type
        data = self.data_splits.get(split)
        if data is None:
            raise ValueError(f"Data split '{split}' not found.")
        
        # Randomly select start indices for sequences in the batch
        # Tensor of shape (batch_size,) with random sequence start indices between 0 and len(data) - block_size
        ix = torch.randint(len(data) - self.block_size, (batch_size,))

        # Generate the input sequences (x) based on the selected indices
        # Accumulate and add each sequence of this batch to form a tensor
        x = torch.stack([data[i:i+self.block_size] for i in ix])

        # Generate the target sequences (y), which are the input sequences offset by one token
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        return x, y



