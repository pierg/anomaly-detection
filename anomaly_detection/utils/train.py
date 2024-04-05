"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
from torch import nn
from trainers.trainer import Trainer

from anomaly_detection.data.text_data import TextBatchGenerator
from anomaly_detection.optimizers.adamw import AdamW


def generate(
    model: nn.Module, context: torch.Tensor, tokenizer, max_new_tokens: int = 500
):
    print("Generating text...")
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[
            0
        ].tolist()
    return tokenizer.decode(generated_tokens)


def train_model(
    model: nn.Module, hyperparameters: dict, batch_generator: TextBatchGenerator
):
    print("Training the model...")
    optimizer = AdamW(model.parameters(), lr=hyperparameters["learning_rate"])
    loss_fn = torch.nn.functional.cross_entropy
    trainer = Trainer(
        data=batch_generator, model=model, optimizer=optimizer, loss_fn=loss_fn
    )
    trainer.train(
        hyperparameters["batch_size"],
        hyperparameters["max_iters"],
        hyperparameters["eval_interval"],
    )
