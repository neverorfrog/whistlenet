import torch
from omegaconf import OmegaConf
from torch.nn.utils.parametrizations import weight_norm

from ckconv.customlayers import Linear1d
from ckconv.expression import Multiply, Sine, Swish


class KernelNet(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        activation: str,
        bias: bool,
        omega_0: float,
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

        # Linear = Linear[config.linear_type]
        # dim_linear = config.dim_linear
        # Norm = Norm[config.norm_type]
        Activation = {
            "ReLU": torch.nn.ReLU,
            "LeakyReLU": torch.nn.LeakyReLU,
            "Swish": Swish,
            "Sine": Sine,
        }[activation]
        Linear = (
            Linear1d  # Implements a Linear layer in terms of 1x1 Convolutions.
        )
        Mul = Multiply  # Multiplies the input by a constant

        # The input of the network is a vector of relative positions and so input_dim = 1
        self.kernel_net = torch.nn.Sequential(
            # 1st layer
            weight_norm(Linear(1, hidden_channels, bias=bias)),
            Mul(omega_0),
            Activation(),
            # 2nd Layer
            weight_norm(Linear(hidden_channels, hidden_channels, bias=bias)),
            Mul(omega_0),
            Activation(),
            # 3rd Layer
            weight_norm(Linear(hidden_channels, out_channels, bias=bias)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_net(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
