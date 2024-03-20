"""
Author: Piergiuseppe Mallozzi
Date: 2024
"""

import torch
import torch.nn as nn
from loguru import logger

class DeepLog(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialize the DeepLog model.
        """
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        # Log input tensor size
        logger.info(f'Input tensor size: {x.size()}')
        logger.info(f'Reshaping..')

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        logger.info(f'After LSTM layer, tensor size: {out.size()}')
        
        # Pass the output from the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        logger.info(f'After fully connected layer, tensor size: {out.size()}')
        
        return out
