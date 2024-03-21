"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
from torch.utils.data import DataLoader

def print_dataloader_info(dataloader: DataLoader):
    """
    Prints information about batches from a DataLoader object, including shapes, types,
    and details for the first few batches.

    :param dataloader: The DataLoader object to print information for.
    """
    print("========== DataLoader Information ==========\n")
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        print(f"=== Batch {i + 1} ===")
        if i == 0:
            print_batch_info(input_tensor, target_tensor)
            pretty_print_tensor(input_tensor, "Input Tensor", num_entries=2)
            pretty_print_tensor(target_tensor, "Target Tensor", num_entries=2)
        else:
            pretty_print_tensor_info(input_tensor, f"Element {i + 1} Input Tensor")
            pretty_print_tensor_info(target_tensor, f"Element {i + 1} Target Tensor")
        
        if i == 1:  # Limit to first 2 batches for brevity
            break

    print("DataLoader information printing complete.\n")
    
def print_batch_info(input_tensor: torch.Tensor, target_tensor: torch.Tensor):
    """
    Prints information about the batch including the shape of the input and target tensors,
    and details of the first batch.
    """
    print("  Input tensor shape: ", input_tensor.shape)
    print("  Target tensor shape:", target_tensor.shape)
    print("\n  Details of the first batch:")
    for b in range(input_tensor.shape[0]):  # Adjust based on your data
        context = input_tensor[b, :]
        target = target_tensor[b] if target_tensor.dim() == 1 else target_tensor[b, :]
        print(f"    Element {b}: Context: {context.tolist()}, Target: {target.tolist() if target_tensor.dim() == 1 else target.item()}")

def pretty_print_tensor(tensor: torch.Tensor, name: str = "Tensor", num_entries: int = 2):
    """
    Pretty prints information about a PyTorch tensor.
    """
    print(f"    --- {name} Information ---")
    print(f"    Shape: {tensor.shape}  Datatype: {tensor.dtype}")
    print(f"    Data: {tensor.tolist()[:num_entries]}...")

def pretty_print_tensor_info(tensor: torch.Tensor, name: str = "Tensor"):
    """
    Pretty prints summary information about a PyTorch tensor.
    """
    print(f"    --- {name} Info ---")
    print(f"    Shape: {tensor.shape}  Datatype: {tensor.dtype}\n")

