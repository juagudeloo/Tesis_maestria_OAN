import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int):
        super().__init__()
        self.name = "SimpleLinear"
        self.simple_model = nn.Sequential(
        nn.Linear(in_features = in_shape, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = hidden_units),
        nn.ReLU(),
        nn.Linear(in_features = hidden_units, out_features = out_shape)
        )
    def forward(self, x):
        return self.simple_model(x)

class CNN1DModel(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int, signal_length: int):
        super().__init__()
        self.name = "SimpleCNN1D"
        
        #Hyperparameters
        padding = 1
        kernel_size = 2
        stride = 1
        conv_out_size = int((signal_length+2*padding-kernel_size)/stride + 1)
        
        #Model
        self.simple_conv = nn.Sequential(
        nn.Conv1d(in_channels=in_shape, out_channels=hidden_units, kernel_size = kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units*2, kernel_size = kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=hidden_units*2, out_channels=hidden_units*4, kernel_size = kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv1d(in_channels=hidden_units*4, out_channels=hidden_units*8, kernel_size = kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features = (hidden_units*8)*conv_out_size, out_features = out_shape)
        )
    def forward(self, x):
        return self.simple_conv(x)