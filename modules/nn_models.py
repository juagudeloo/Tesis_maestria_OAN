import torch
from torch import nn


class ThermodynamicModel(nn.Module):
    def __init__(self):

        super(ThermodynamicModel, self).__init__()
        self.name = "ThermodynamicModel"


class MagneticFieldModel(nn.Module):

    def __init__(self):

        super(MagneticFieldModel, self).__init__()
        self.name = "MagneticFieldModel"
        





####################################
# Aside blocks
####################################

class LineaBlock(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int):
        super().__init__()
        self.name = "SimpleLinear"
        self.simple_model = nn.Sequential(
        nn.Linear(in_features = in_shape, out_features = hidden_units),
        nn.ReLU()
        )
    def forward(self, x):
        return self.simple_model(x)


class CNN1DBlock(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_channels: int, signal_length: int):
        super().__init__()
        self.name = "SimpleCNN1D"
        
        # Hyperparameters
        padding = 1
        kernel_size = 2
        stride = 1
        
        # Model
        self.simple_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_shape, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.simple_conv(x)