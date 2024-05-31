import torch
from omegaconf import OmegaConf
from torch.nn.utils.parametrizations import weight_norm

import core
from core.ckconv.config import ActivationFunction, Linear, Norm


class KernelNet(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        bias: bool,
        omega_0: float,
    ):
        """
        Creates an 3-layer MLP, which parameterizes a convolutional kernel as:

        relative position -> hidden_channels -> hidden_channels -> in_channels * out_channels

        :param out_channels: output channels of the resulting convolutional kernel.
        :param hidden_channels: Number of hidden units per hidden layer.
        :param n_layers: Number of layers.
        :param activation_function: Activation function used.
        :param norm_type: Normalization type used.
        :param bias:  If True, adds a learnable bias to the layers.
        :param weight_dropout: Dropout rate applied to the sampled convolutional kernel.
        """
        super().__init__()

        # Linear = Linear[config.linear_type]
        # dim_linear = config.dim_linear
        # Norm = Norm[config.norm_type]
        # ActivationFunction = ActivationFunction[config.activation_function]

        ActivationFunction = core.ckconv.Sine
        Linear = (
            core.ckconv.Linear1d
        )  # Implements a Linear layer in terms of 1x1 Convolutions.
        Multiply = core.ckconv.Multiply  # Multiplies the input by a constant

        # The input of the network is a vector of relative positions. That is, input_dimension = 1.
        self.kernel_net = torch.nn.Sequential(
            # 1st layer
            weight_norm(Linear(1, hidden_channels, bias=bias)),
            Multiply(omega_0),
            ActivationFunction(),
            # 2nd Layer
            weight_norm(Linear(hidden_channels, hidden_channels, bias=bias)),
            Multiply(omega_0),
            ActivationFunction(),
            # 3rd Layer
            weight_norm(Linear(hidden_channels, out_channels, bias=bias)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_net(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
