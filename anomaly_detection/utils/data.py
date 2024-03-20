"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

from pathlib import Path
import torch

from data.text_data import CharacterTokenizer, TextBatchGenerator, TextDataLoader


def split_data(data: torch.Tensor, train_val_split: float = 0.9) -> tuple:
    """
    Splits the data tensor into training and validation sets
    """
    n = int(train_val_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data



def process_data(data_path: Path, hyperparameters: dict):
    # Load and Tokenize Data
    data_loader = TextDataLoader()
    raw_text = data_loader.load_data(data_path)
    tokenizer = CharacterTokenizer(raw_text)
    tokenized_text = tokenizer.tokenize(raw_text)

    # Split Data
    train_data, val_data = split_data(tokenized_text, train_val_split=hyperparameters["train_val_split"])

    # Initialize Batch Generator
    data_splits = {'train': train_data, 'val': val_data}
    batch_generator = TextBatchGenerator(data_splits, block_size=hyperparameters["block_size"])

    vocab_size = tokenizer.vocab_size()
    return vocab_size, tokenizer, batch_generator