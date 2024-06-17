import torch
from omegaconf import OmegaConf

from ckconv.ckconv import CKConv
from ckconv.customlayers import LayerNorm, Linear1d


class CKBlock(torch.nn.Module):
    """
    Implements a Residual Block with a convolutional kernel parametrized by a
    neural network.

    input
    | ---------------|
    CKConv           |
    LayerNorm        |
    ReLU             |
    DropOut          |
    |                |
    CKConv           |
    LayerNorm        |
    ReLU             |
    DropOut          |
    + <--------------|
    |
    ReLU
    |
    output

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        config (OmegaConf): The config object containing the parameters for the
            convolutional kernel.

    Returns:
        torch.nn.Module: The CKBlock module.
    """

    def __init__(self, in_channels: int, out_channels: int, config: OmegaConf):
        super().__init__()

        self.conv1 = CKConv(in_channels, out_channels, config)
        self.conv2 = CKConv(out_channels, out_channels, config)

        # Normalization layers
        self.norm1 = LayerNorm(out_channels)
        self.norm2 = LayerNorm(out_channels)

        # Dropout layer
        self.dropout: torch.nn.Module = torch.nn.Dropout(
            p=config.network.dropout
        )

        # Residual connection
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(Linear1d(in_channels, out_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.dropout(torch.relu(self.norm1(self.conv1(x))))
        out = torch.relu(
            self.dropout(torch.relu(self.norm2(self.conv2(out)))) + shortcut
        )
        return out
