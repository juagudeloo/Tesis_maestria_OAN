import torch
from torch import nn

######################################################################################################
# MSCNN
######################################################################################################

class MSCNN(nn.Module):
    def __init__(self, scales: list[int], nwl_points: int, in_channels: int = 4, c1_filters: int = 16, c2_filters: int = 32, kernel_size: int = 5, stride: int = 1, padding: int = 0, pool_size: int = 2, n_outputs: int = 5):
        super(MSCNN, self).__init__()
        # 1. Multi-scale feature mapping
        self.multi_scale_feature_mapping = MultiScaleFeatureMapping(scales=scales, in_channels=in_channels, c1_filters=c1_filters, c2_filters=c2_filters, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size)
        # 2. Flatten layer
        self.flatten_layer = nn.Flatten()
        # 3. Flatten size calculation layers
        def get_output_shape(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
            return (input_size - kernel_size + 2 * padding) // stride + 1
        pool_stride = 2
        total_features = 0
        for s in scales:
            conv_output_shapes = nwl_points // s
            for _ in range(2):
                conv_output_shapes = get_output_shape(conv_output_shapes, kernel_size, stride=1, padding=0)//pool_stride 
            total_features += conv_output_shapes
        flatten_size = total_features * c2_filters
        # 4. Linear layers
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features = flatten_size, out_features = flatten_size),
            nn.ReLU())
        self.linear_layers = nn.ModuleList([self.linear_layer for _ in range(4)])
        # 5. Output layer
        self.output_layer = nn.Linear(in_features = flatten_size, out_features = n_outputs)

    def forward(self, x):
        x = self.multi_scale_feature_mapping(x)
        x = self.flatten_layer(x)
        for layer in self.linear_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

#-----------------------------------------------------------------------------
# Helper classes
#-----------------------------------------------------------------------------

class MultiScaleFeatureMapping(nn.Module):
    def __init__(self, scales: list[int], in_channels: int = 4, c1_filters: int = 16, c2_filters: int = 32, kernel_size: int = 5, stride: int = 1, padding: int = 0, pool_size: int = 2):
        super(MultiScaleFeatureMapping, self).__init__()
        self.scales = scales
        self.c2_filters = c2_filters
        self.coarse_grains = nn.ModuleDict({f"scale_{s}": CoarseGrain(scale = s) for s in scales})
        self.conv_block = ConvBlock(in_channels=in_channels, c1_filters=c1_filters, c2_filters=c2_filters, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size)
        
    def forward(self, x):
        feature_maps = torch.empty((x.size(0), self.c2_filters, 0), dtype=torch.float32, device=x.device)
        for s in self.scales:
            stokes_cg = self.coarse_grains[f"scale_{s}"](x)
            conv_block_output = self.conv_block(stokes_cg)
            feature_maps = torch.cat((feature_maps, conv_block_output), dim=-1)
        return feature_maps
    
class CoarseGrain(nn.Module):
    def __init__(self, scale: int):
        super(CoarseGrain, self).__init__()
        self.scale = scale

    def forward(self, x):
        unfolded = x.unfold(dimension=2, size=self.scale, step=self.scale)
        stokes_scale = unfolded.mean(dim=-1)
        
        return stokes_scale
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 4, c1_filters: int = 16, c2_filters: int = 32, kernel_size: int = 5, stride: int = 1, padding: int = 0, pool_size: int = 2):
        super(ConvBlock, self).__init__()
        self.c1 = nn.Sequential(nn.Conv1d(in_channels = in_channels, out_channels=c1_filters, kernel_size=kernel_size, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=pool_size))
        self.c2 = nn.Sequential(nn.Conv1d(in_channels = c1_filters, out_channels=c2_filters, kernel_size=kernel_size, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=pool_size))

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x
    

######################################################################################################
# CNN1D_4CHANNELS
######################################################################################################

class CNN1D(nn.Module):
    def __init__(self, in_shape: int, out_shape: int, hidden_units: int, signal_length: int):
        super().__init__()
        self.name = "SimpleCNN1D"
        
        # Hyperparameters
        padding = 1
        kernel_size = 2
        stride = 1
        
        # Model
        # Calculate the output length after each Conv1d layer
        def calculate_conv_output_length(input_length, kernel_size, stride, padding):
            return ((input_length - kernel_size + 2 * padding) // stride) + 1

        conv1_output_length = calculate_conv_output_length(signal_length, kernel_size, stride, padding)
        conv2_output_length = calculate_conv_output_length(conv1_output_length, kernel_size, stride, padding)
        conv3_output_length = calculate_conv_output_length(conv2_output_length, kernel_size, stride, padding)
        conv4_output_length = calculate_conv_output_length(conv3_output_length, kernel_size, stride, padding)
        conv5_output_length = calculate_conv_output_length(conv4_output_length, kernel_size, stride, padding)
        
        # Model
        self.simple_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_shape, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=hidden_units * conv5_output_length, out_features=out_shape)
        )
    
    def forward(self, x):
        return self.simple_conv(x)