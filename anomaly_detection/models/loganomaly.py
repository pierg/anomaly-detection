import torch
import torch.nn as nn
from loguru import logger
from anomaly_detection.models.modules.head import Head

class LogAnomaly(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_keys: int, head_size: int, n_embd: int, block_size: int, dropout: float = 0.1):
        """
        Initializes the LogAnomaly model with specific configurations for its layers.
        """
        super(LogAnomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Initialize LSTM layers
        logger.info(f'Initializing LSTM0 layer with input size: {input_size}, hidden size: {hidden_size}, num layers: {num_layers}')
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        logger.info(f'Initializing LSTM1 layer with input size: {input_size}, hidden size: {hidden_size}, num layers: {num_layers}')
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Initialize self-attention Heads
        self.head0 = Head(head_size, n_embd, block_size, dropout)
        self.head1 = Head(head_size, n_embd, block_size, dropout)

        # Initialize the final Linear layer
        logger.info(f'Initializing Linear layer with input size: {2 * head_size}, output size: {num_keys}')
        self.fc = nn.Linear(2 * head_size, num_keys)
    
    def forward(self, features: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Defines the forward pass of the LogAnomaly model.
        """
        input0, input1 = features

        # Prepare the hidden and cell states
        h0_0, c0_0 = self.init_hidden(input0.size(0), device)
        h0_1, c0_1 = self.init_hidden(input1.size(0), device)
        
        # Process inputs through LSTM layers
        out0, _ = self.lstm0(input0, (h0_0, c0_0))
        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        
        # Apply attention to LSTM outputs
        attn_out0 = self.head0(out0)
        attn_out1 = self.head1(out1)
        
        # Concatenate attention outputs and pass through the final Linear layer
        multi_out = torch.cat((attn_out0, attn_out1), dim=-1)
        out = self.fc(multi_out)
        
        logger.debug(f'Processed features through LogAnomaly: {multi_out.shape}')
        return out

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes hidden and cell states for LSTM layers.
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        logger.debug(f'Initialized hidden and cell states: {h.shape}, {c.shape}')
        return h, c
