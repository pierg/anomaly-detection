"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from loguru import logger

class Trainer:
    """
    Trainer class for training a model using a specified optimizer.
    Handles the training loop, loss calculation, and evaluation.
    """

    def __init__(self, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Initializes the Trainer with training and validation data loaders, model, optimizer, and loss function.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, max_iters: int, eval_interval: int):
        """
        Trains the model for a specified number of iterations.
        Evaluates the model at regular intervals.
        """
        iter_count = 0
        while iter_count < max_iters:
            for xb, yb in self.train_loader:
                if iter_count >= max_iters:
                    break

                # Log the sizes of the input and target tensors
                logger.info(f'Input batch size: {xb.size()}')
                logger.info(f'Target batch size: {yb.size()}')

                # Forward pass: compute predictions from model
                logits = self.model(xb)
                
                # Log the size of the output tensor
                logger.info(f'Output logits size: {logits.size()}')

                # Assuming your model's output and targets are already shaped correctly
                loss = self.loss_fn(logits, yb)

                # Backward pass: compute gradients and update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Periodically evaluate the model and print the losses
                if iter_count % eval_interval == 0:
                    train_loss = loss.item()  # Current batch loss as an approximation
                    val_loss = self.evaluate_loss()
                    print(f'Iter {iter_count:4d} | Train Loss {train_loss:6.4f} | Val Loss {val_loss:6.4f}')

                iter_count += 1

    @torch.no_grad() # Disable gradient calculation for this function
    def evaluate_loss(self) -> float:
        """
        Evaluates the model's performance on the validation set.
        Returns the average loss.
        """
        self.model.eval() # Set model to evaluation mode
        total_loss = 0
        total_count = 0
        
        for xb, yb in self.val_loader:
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)  # Multiply by batch size
            total_count += xb.size(0)
        
        avg_loss = total_loss / total_count
        
        self.model.train() # Set model back to training mode
        return avg_loss
