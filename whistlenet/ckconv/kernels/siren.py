from math import sqrt

import torch

import whistlenet.core.layers as layers
from config import KernelConfig
from whistlenet.core.utils import getcallable


class Siren(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, config: KernelConfig
    ):
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

        Activation: torch.nn.Module = getcallable(layers, config.activation)
        Linear: torch.nn.Module = getcallable(layers, config.linear_type)

        # The input of the network is a vector of relative positions and so input_dim = 1
        self.layers = []
        self.layers.extend(
            torch.nn.Sequential(
                Linear(1, config.hidden_channels, bias=config.bias),
                Activation(),
                layers.Multiply(config.omega_0),
            )
        )

        self.layers.extend(
            torch.nn.Sequential(
                Linear(
                    config.hidden_channels,
                    config.hidden_channels,
                    bias=config.bias,
                ),
                Activation(),
                layers.Multiply(config.omega_0),
            )
        )

        self.output_linear = Linear(
            config.hidden_channels, out_channels, bias=config.bias
        )
        self.layers.extend(
            torch.nn.Sequential(self.output_linear, Activation())
        )

        self.kernel_net = torch.nn.Sequential(*self.layers)
        self.initialize(config.omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_net(x)

    def initialize(self, omega_0):
        net_layer = 1
        for i, m in enumerate(self.kernel_net.modules()):
            if isinstance(
                m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)
            ):
                if net_layer == 1:
                    w_std = 1 / m.weight.shape[1]
                    m.weight.data.uniform_(
                        -w_std, w_std
                    )  # Normally (-1, 1) / in_dim but we only use 1D inputs.
                    # Important! Bias is not defined in original SIREN implementation!
                    net_layer += 1
                else:
                    w_std = sqrt(6.0 / m.weight.shape[1]) / omega_0
                    m.weight.data.uniform_(
                        -w_std,
                        # the in_size is dim 2 in the weights of Linear and Conv layers
                        w_std,
                    )
                # TODO: Important! Bias is not defined in original SIREN implementation
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        # The final layer must be initialized differently because it is not multiplied by omega_0
        torch.nn.init.kaiming_uniform_(
            self.output_linear.weight, nonlinearity="linear"
        )
        if self.output_linear.bias is not None:
            self.output_linear.bias.data.fill_(0.0)
