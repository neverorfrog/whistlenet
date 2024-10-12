import torch
import torch.nn as nn

from config import WhistlenetConfig
from whistlenet.ckconv.layers.ckconv import CKConv
from whistlenet.core.layers import Linear1d


class CKBlock(torch.nn.Module):
    """
    Implements a Residual Block with a convolutional kernel parametrized by a
    neural network.

    input
    | ---------------|
    BatchNorm        |
    CKConv           |
    GELU             |
    DropOut          |
    Linear           |
    GELU             |
    + <--------------|
    |
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

    def __init__(
        self, in_channels: int, out_channels: int, config: WhistlenetConfig
    ):
        super().__init__()

        # Activation
        self.activation = nn.ELU()

        # Continuous convolution layer
        self.conv = CKConv(in_channels, out_channels, config)

        # Normalization layer
        self.batch_norm = nn.BatchNorm1d(in_channels)

        # Dropout layer
        self.dropout: torch.nn.Module = torch.nn.Dropout(p=config.dropout)

        self.layers = nn.Sequential(
            self.batch_norm,
            self.conv,
            self.activation,
            self.dropout,
            Linear1d(out_channels, out_channels),
            self.activation,
        )

        # Residual connection
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(Linear1d(in_channels, out_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out: torch.Tensor = self.layers(x)
        shortcut = self.shortcut(x)[:, :, : out.size(2)]
        out = self.activation(out + shortcut)

        return out
