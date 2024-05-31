import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool1dSame(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool1dSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        padding = (self.kernel_size - 1) // 2
        x = F.pad(x, (padding, padding), mode="constant", value=0)
        return F.max_pool1d(x, self.kernel_size, self.stride)


def Linear1d(
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = True
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
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=bias
    )
