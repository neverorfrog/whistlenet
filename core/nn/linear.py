import torch
import torch.nn as nn

def Linear1d(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    bias: bool = True
) -> torch.nn.Module:
    """
    Create a linear layer in terms of pointwise convolution.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int, optional): The stride of the convolution. Defaults to 1.
        bias (bool, optional): If True, adds a bias term to the convolution. Defaults to True.

    Returns:
        torch.nn.Module: The created 1D linear convolutional layer.
    """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)