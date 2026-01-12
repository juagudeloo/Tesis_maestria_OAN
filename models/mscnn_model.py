import torch
import torch.nn as nn


def _conv1d_output_length(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
	"""Compute the output length of a 1D conv layer."""
	return (input_size - kernel_size + 2 * padding) // stride + 1


class CoarseGrain(nn.Module):
	def __init__(self, scale: int) -> None:
		super().__init__()
		self.scale = scale

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		unfolded = x.unfold(dimension=2, size=self.scale, step=self.scale)
		return unfolded.mean(dim=-1)


class ConvBlock(nn.Module):
	def __init__(
		self,
		*,
		in_channels: int = 2,
		c1_filters: int = 16,
		c2_filters: int = 32,
		kernel_size: int = 5,
		stride: int = 1,
		padding: int = 0,
		pool_size: int = 2,
	) -> None:
		super().__init__()
		self.c1 = nn.Sequential(
			nn.Conv1d(
				in_channels=in_channels,
				out_channels=c1_filters,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
			),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=pool_size),
		)
		self.c2 = nn.Sequential(
			nn.Conv1d(
				in_channels=c1_filters,
				out_channels=c2_filters,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
			),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=pool_size),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.c1(x)
		x = self.c2(x)
		return x


class MultiScaleFeatureMapping(nn.Module):
	def __init__(
		self,
		*,
		scales: list[int],
		in_channels: int = 2,
		c1_filters: int = 16,
		c2_filters: int = 32,
		kernel_size: int = 5,
		stride: int = 1,
		padding: int = 0,
		pool_size: int = 2,
	) -> None:
		super().__init__()
		self.scales = scales
		self.c2_filters = c2_filters
		self.coarse_grains = nn.ModuleDict({f"scale_{s}": CoarseGrain(scale=s) for s in scales})
		self.conv_block = ConvBlock(
			in_channels=in_channels,
			c1_filters=c1_filters,
			c2_filters=c2_filters,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			pool_size=pool_size,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feature_maps = torch.empty((x.size(0), self.c2_filters, 0), dtype=torch.float32, device=x.device)
		for s in self.scales:
			stokes_cg = self.coarse_grains[f"scale_{s}"](x)
			conv_block_output = self.conv_block(stokes_cg)
			feature_maps = torch.cat((feature_maps, conv_block_output), dim=-1)
		return feature_maps


class LinearBlock(nn.Module):
    def __init__(
        self, 
        *, 
        in_features: int, 
        out_features: int, 
        n_layers: int = 4,
        dropout_rate: float = 0.0  # Add dropout parameter
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
                )
                for _ in range(n_layers)
            ]
        )
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class MSCNNInversionModel(nn.Module):
    def __init__(
        self,
        *,
        scales: list[int],
        in_channels: int = 2,
        c1_filters: int = 16,
        c2_filters: int = 32,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
        n_linear_layers: int = 4,
        output_features: int = 3 * 21,
        input_length: int = 112,
        dropout_rate: float = 0.2,  # Add dropout parameter
    ) -> None:
        super().__init__()

        self.msfm = MultiScaleFeatureMapping(
            scales=scales,
            in_channels=in_channels,
            c1_filters=c1_filters,
            c2_filters=c2_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_size=pool_size,
        )

        # Compute flatten size from expected input length.
        total_features = 0
        for s in scales:
            length = input_length // s
            length = _conv1d_output_length(length, kernel_size, stride=stride, padding=padding)
            length = length // pool_size
            length = _conv1d_output_length(length, kernel_size, stride=stride, padding=padding)
            length = length // pool_size
            total_features += length
        flatten_size = total_features * c2_filters

        self.flatten = nn.Flatten()
        self.linear_block = LinearBlock(
            in_features=flatten_size,
            out_features=output_features,
            n_layers=n_linear_layers,
            dropout_rate=dropout_rate,  # Pass dropout to LinearBlock
        )

    def forward(self, x: torch.Tensor, stocastic_generation: bool = False, stocastic_steps: int = 30) -> torch.Tensor:
        x = self.msfm(x)
        x = self.flatten(x)

        if stocastic_generation and not self.training:
            # Enable dropout during inference for Monte Carlo sampling
            self.linear_block.train()  # Put linear block in training mode
            
            predictions = []
            for _ in range(stocastic_steps):  # 30 stochastic forward passes
                pred = self.linear_block(x)
                predictions.append(pred.unsqueeze(0))
            
            self.linear_block.eval()  # Restore eval mode
            
            predictions = torch.cat(predictions, dim=0)  # (stocastic_steps, batch, 63)
            
            return predictions

        return self.linear_block(x)


__all__ = [
	"CoarseGrain",
	"ConvBlock",
	"MultiScaleFeatureMapping",
	"LinearBlock",
	"MSCNNInversionModel",
]

"""
import torch
from models.mscnn_model import MSCNNInversionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate with your preferred scales and expected input length
model = MSCNNInversionModel(scales=[1, 2, 3], input_length=112).to(device)

# Dummy batch of Stokes profiles: shape (batch, channels=2, length=112)
dummy_input = torch.randn(4, 2, 112, device=device)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)  # should be (4, 63) for 3*21 outputs
"""