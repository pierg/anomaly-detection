import torch
import torch.nn as nn
from loguru import logger

class DeepLog(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int):
        """
        Initialize the DeepLog model.
        """
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Log layer dimensions upon initialization
        logger.info(f'Initializing LSTM layer with input size: {input_size}, hidden size: {hidden_size}, num layers: {num_layers}')
        # Initializing LSTM layer with input size: 1, hidden size: 64, num layers: 2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        logger.info(f'Initializing Linear layer with input size: {hidden_size}, output size: {output_size}')
        # Initializing Linear layer with input size: 64, output size: 28
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        # Log input tensor size
        x = x.float()
        x = x.unsqueeze(-1)
        
        logger.debug(f'xin:\t{x.size()}\t{x.dtype}')
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        logger.debug(f"x shape: {x.shape}, h0 shape: {h0.shape}, c0 shape: {c0.shape}")
        
        # Log initialization of hidden and cell states
        logger.debug(f'h0 :\t{h0.size()}\t{h0.dtype}')
        logger.debug(f'c0 :\t{c0.size()}\t{c0.dtype}')

        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        logger.debug(f'out:\t{out.size()}\t{out.dtype}')
        
        # Pass the output from the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])
        logger.debug(f'fc:\t{out.size()}\t{out.dtype}')
        
        return out
