import torch
import torch.nn as nn

class CoarseGrain(nn.Module):
    def __init__(self, scale: int):
        super(CoarseGrain, self).__init__()
        self.scale = scale

    def forward(self, x):
        unfolded = x.unfold(dimension=2, size=self.scale, step=self.scale)
        stokes_scale = unfolded.mean(dim=-1)
        return stokes_scale

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 2, c1_filters: int = 16, c2_filters: int = 32, kernel_size: int = 5, stride: int = 1, padding: int = 0, pool_size: int = 2):
        super(ConvBlock, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=c1_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=c1_filters, out_channels=c2_filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x

class MultiScaleFeatureMapping(nn.Module):
    def __init__(self, scales: list[int], in_channels: int = 2, c1_filters: int = 16, c2_filters: int = 32, kernel_size: int = 5, stride: int = 1, padding: int = 0, pool_size: int = 2):
        super(MultiScaleFeatureMapping, self).__init__()
        self.scales = scales
        self.c2_filters = c2_filters
        self.coarse_grains = nn.ModuleDict({f"scale_{s}": CoarseGrain(scale=s) for s in scales})
        self.conv_block = ConvBlock(in_channels=in_channels, c1_filters=c1_filters, c2_filters=c2_filters, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size)

    def forward(self, x):
        feature_maps = torch.empty((x.size(0), self.c2_filters, 0), dtype=torch.float32, device=x.device)
        for s in self.scales:
            stokes_cg = self.coarse_grains[f"scale_{s}"](x)
            conv_block_output = self.conv_block(stokes_cg)
            feature_maps = torch.cat((feature_maps, conv_block_output), dim=-1)
        return feature_maps

class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_layers: int = 4):
        super(LinearBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features),
                nn.ReLU()
            ) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def get_output_shape(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return (input_size - kernel_size + 2 * padding) // stride + 1

class MSCNNInversionModel(nn.Module):
    def __init__(self, scales: list[int], in_channels: int = 2, c1_filters: int = 16, c2_filters: int = 32, kernel_size: int = 5, stride: int = 1, padding: int = 0, pool_size: int = 2, n_linear_layers: int = 4, output_features: int = 3*21, input_length: int = 112):
        super(MSCNNInversionModel, self).__init__()
        self.msfm = MultiScaleFeatureMapping(scales=scales, in_channels=in_channels, c1_filters=c1_filters, c2_filters=c2_filters, kernel_size=kernel_size, stride=stride, padding=padding, pool_size=pool_size)
        total_features = 0
        for s in scales:
            conv_output_shapes = input_length // s
            for _ in range(2):
                conv_output_shapes = get_output_shape(conv_output_shapes, kernel_size, stride=stride, padding=padding)//pool_size
            total_features += conv_output_shapes
        flatten_size = total_features * c2_filters
        self.flatten = nn.Flatten()
        self.linear_block = LinearBlock(in_features=flatten_size, out_features=output_features, n_layers=n_linear_layers)

    def forward(self, x):
        x = self.msfm(x)
        x = self.flatten(x)
        x = self.linear_block(x)
        return x
