import torch
from torch import nn

class SimpleLinearModel(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int):
        super().__init__()
        self.name = "SimpleLinear"
        self.simple_model = nn.Sequential(
        nn.Linear(in_features = in_shape, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = out_shape)
        )
    def forward(self, x):
        return self.simple_model(x)


class SimpleCNN1DModel(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int, signal_length: int):
        super().__init__()
        self.name = "SimpleCNN1D"
        
        # Hyperparameters
        padding = 1
        kernel_size = 2
        stride = 1
        
        # Model
        self.simple_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_shape, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Ensure the output size is fixed to 1
            nn.Flatten(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=hidden_units, out_features=out_shape)
        )
    
    def forward(self, x):
        return self.simple_conv(x)