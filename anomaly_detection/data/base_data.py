"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class DataLoader(ABC):
    """Abstract class for data loading."""

    @abstractmethod
    def load_data(self, file_path: str) -> str:
        """Load data from a specified file path."""
        pass


class Tokenizer(ABC):
    """Abstract class for tokenization."""

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize given text."""
        pass

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tokens back to text."""
        pass


class BatchGenerator(ABC):
    """Abstract class for generating data batches."""

    @abstractmethod
    def get_batch(
        self, split: str, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of data for a specified split and batch size."""
        pass
