import torch
from torch.nn.utils.parametrizations import weight_norm

import whistlenet.core.layers as layers
from config import KernelConfig
from whistlenet.core.utils import getcallable


class KernelNet(torch.nn.Module):
    def __init__(self, out_channels: int, config: KernelConfig):
        """
        Creates an 3-layer MLP that parameterizes a convolutional kernel as:

        relative position (1) -> hidden_channels (32) -> hidden_channels (32) -> out_channels (1) * in_channels (1)

        Args:
            out_channels (int): The number of output channels.
            hidden_channels (int): The number of hidden channels.
            activation (str): The activation function to use.
            bias (bool): If True, adds a bias term to the convolution.
            omega_0 (float): The initial value of the kernel..

        Returns:
            torch.nn.Module: The created 3-layer MLP.
        """
        super().__init__()

        # dim_linear = config.dim_linear
        # Norm = Norm[config.norm_type]

        Activation: torch.nn.Module = getcallable(layers, config.activation)
        Linear: torch.nn.Module = getcallable(layers, config.linear_type)

        # The input of the network is a vector of relative positions and so input_dim = 1
        self.kernel_net = torch.nn.Sequential(
            # 1st layer
            weight_norm(Linear(1, config.hidden_channels, bias=config.bias)),
            layers.Multiply(config.omega_0),
            Activation(),
            # 2nd Layer
            weight_norm(
                Linear(
                    config.hidden_channels,
                    config.hidden_channels,
                    bias=config.bias,
                )
            ),
            layers.Multiply(config.omega_0),
            Activation(),
            # 3rd Layer
            weight_norm(
                Linear(config.hidden_channels, out_channels, bias=config.bias)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_net(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
