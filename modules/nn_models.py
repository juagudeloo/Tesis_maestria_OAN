import torch
from torch import nn

class SimpleLinearModel(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_units):
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
    def __init__(self, in_shape, out_shape, hidden_units):
        super().__init__()
        self.name = "SimpleCNN1D"
        padding = 1
        self.simple_conv = nn.Sequential(
        nn.Conv1d(in_channels=in_shape, out_channels=hidden_units, kernel_size = 2, stride=1, padding=padding),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features = hidden_units*5, out_features = out_shape)
        )
    def forward(self, x):
        return self.simple_conv(x)