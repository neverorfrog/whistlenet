import torch
import torch.nn as nn

from ckconv.util import causal_fftconv


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


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-12,
    ):
        """Uses GroupNorm implementation with group=1 for speed."""
        super().__init__()
        # we use GroupNorm to implement this efficiently and fast.
        self.layer_norm = torch.nn.GroupNorm(
            1, num_channels=num_channels, eps=eps
        )

    def forward(self, x):
        return self.layer_norm(x)


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        weight_dropout: float,
    ):
        """
        Applies a 1D convolution over an input signal of input_channels.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param bias: If True, adds a learnable bias to the output.
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernels.
        """
        super().__init__()
        self.weight_dropout = weight_dropout
        self.w_dropout = torch.nn.Dropout(p=weight_dropout)

        self.weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size)
        )
        self.weight.data.normal_(0, 0.01)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)

    def forward(self, x):

        # Dropout weight values if required
        if self.weight_dropout != 0.0:
            weight = self.w_dropout(self.weight)
        else:
            weight = self.weight

        # Perform causal convolution
        return causal_fftconv(
            x,
            weight,
            self.bias,
            double_precision=False,
        )
