"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """
    Trainer class for training a model using a specified optimizer.
    Handles the training loop, loss calculation, evaluation, logging, and checkpointing.
    Automatically resumes training from the latest checkpoint if available.
    """

    def __init__(self, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader,
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 checkpoint_dir: str,
                 log_dir: str):
        """
        Initializes the Trainer with training and validation data loaders, model, optimizer, and loss function.
        Also specifies directories for saving checkpoints and logs.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.start_iter = 0  # Default start iteration

    def train(self, max_iters: int, eval_interval: int, checkpoint_interval: int = 1000):
        """
        Trains the model for a specified number of iterations.
        Automatically resumes from the latest checkpoint if available.
        Evaluates the model and saves checkpoints at regular intervals.
        """
        # Check for the latest checkpoint and load it if exists
        self.load_latest_checkpoint()

        logger.info(f'Starting training from iteration {self.start_iter}')

        iter_count = self.start_iter
        while iter_count < max_iters:
            for xb, yb in self.train_loader:
                if iter_count >= max_iters:
                    break

                # Log the sizes of the input and target tensors
                logger.debug(f'xb :\t{xb.size()}\t{xb.dtype}')
                logger.debug(f'yb :\t{yb.size()}\t{yb.dtype}')

                # Forward pass: compute predictions from model
                logits = self.model(xb)
                
                # Log the size of the output tensor
                logger.debug(f'lts:\t{logits.size()}\t{logits.dtype}')
                logger.debug(f'ys :\t{logits.size()}\t{logits.dtype}')

                # Assuming your model's output and targets are already shaped correctly
                loss = self.loss_fn(logits, yb)

                # Backward pass: compute gradients and update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iter_count % eval_interval == 0:
                    train_loss = loss.item()
                    val_loss = self.evaluate_loss()
                    print(f'Iter {iter_count:4d} | Train Loss {train_loss:6.4f} | Val Loss {val_loss:6.4f}')
                    self.writer.add_scalar('Loss/train', train_loss, iter_count)
                    self.writer.add_scalar('Loss/val', val_loss, iter_count)

                if iter_count % checkpoint_interval == 0:
                    self.save_checkpoint(iter_count)

                iter_count += 1
        
        logger.info(f'Training completed after {max_iters} iterations')

    def load_latest_checkpoint(self):
        """
        Loads the latest checkpoint from the checkpoint directory.
        """
        logger.info(f'Looking for latest checkpoint in {self.checkpoint_dir}')
        checkpoints = [ckpt for ckpt in os.listdir(self.checkpoint_dir) if ckpt.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.checkpoint_dir, x)))
            checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_iter = checkpoint.get('iter_count', 0) + 1
            logger.info(f"Resumed from checkpoint: {latest_checkpoint} at iteration {self.start_iter}")



        
        
    def evaluate(self, normal_dataset: DataLoader, abnormal_dataset: DataLoader, num_candidates: int) -> tuple[float, float, float]:
        """
        Evaluate the model using both normal and abnormal datasets.
        Computes metrics such as precision, recall, and F1-measure.
        
        :param normal_dataset: DataLoader for the normal test dataset.
        :param abnormal_dataset: DataLoader for the abnormal test dataset.
        :param num_candidates: Number of top predictions to consider as candidates.
        :return: A tuple containing precision, recall, and F1-measure.
        """

        # Load the latest model checkpoint
        self.load_latest_checkpoint()
        logger.info("Latest model checkpoint loaded for evaluation.")
        
        self.model.eval()  # Ensure model is in evaluation mode


        # Initialize counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Function to process datasets
        def process_dataset(dataset, is_normal):
            nonlocal true_positives, false_positives, false_negatives
            for xb, yb in dataset:
                with torch.no_grad():
                    scores_batch = self.model(xb).cpu().numpy() # Move to CPU and convert to NumPy array for further processing
                    # scores_batch.shape = (batch_size, num_classes)

                    # Sort scores_batch along each row to get sorted indices of predictions
                    sorted_indices = np.argsort(scores_batch, axis=1)

                    # Select the last 'num_candidates' indices for each sample, which are the top predictions
                    # since argsort sorts in ascending order, and the highest scores are considered 'top'.
                    top_predictions = sorted_indices[:, -num_candidates:]

                    # Expand yb to match the shape of top_predictions for element-wise comparison
                    # yb.numpy()[:, None] transforms yb to a 2D column vector, enabling broadcasting when compared with top_predictions
                    expanded_yb = yb.numpy()[:, None]

                    # Check if the true label (expanded_yb) is among the top predictions for each sample
                    # np.any(...) checks each row and returns True if the true label is found among the top predictions, False otherwise.
                    # The result is a boolean array indicating whether each sample's true label was among the top predictions.
                    correct_predictions = np.any(top_predictions == expanded_yb, axis=1)

                    if is_normal:
                        # For normal data, correct predictions are true positives, incorrect are false negatives
                        true_positives += correct_predictions.sum()
                        false_negatives += (~correct_predictions).sum()
                    else:
                        # For abnormal data, incorrect predictions are false positives
                        false_positives += (~correct_predictions).sum()

        # Process normal and abnormal datasets
        process_dataset(normal_dataset, is_normal=True)
        process_dataset(abnormal_dataset, is_normal=False)

        # Compute precision, recall, and F1-measure
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


        logger.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_measure:.4f}')
        
        self.model.train() 

        return precision, recall, f1_measure


    def save_checkpoint(self, iter_count: int):
        """
        Saves a checkpoint of the model's state and optimizer's state.
        Deletes older checkpoints to keep only the latest one.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{iter_count}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iter_count': iter_count,
        }, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')

        # Delete older checkpoints
        for file in os.listdir(self.checkpoint_dir):
            if file != os.path.basename(checkpoint_path) and file.endswith('.pth'):
                os.remove(os.path.join(self.checkpoint_dir, file))
                logger.info(f'Deleted older checkpoint: {file}')


    @torch.no_grad()
    def evaluate_loss(self) -> float:
        """
        Evaluates the model's performance on the validation set.
        Returns the average loss.
        """
        self.model.eval()
        total_loss = 0
        total_count = 0
        
        for xb, yb in self.val_loader:
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
        
        avg_loss = total_loss / total_count
        self.model.train()
        return avg_loss
