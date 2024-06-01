import torch
from torch.nn.utils.parametrizations import weight_norm

from ckconv.expression import Multiply, Sine, Swish
from ckconv.linear import Linear1d

Norm = {
    "BatchNorm": torch.nn.BatchNorm1d,
    "": torch.nn.Identity,
}

ActivationFunction = {
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,
    "Swish": Swish,
    "Sine": Sine,
}

Linear = {1: Linear1d}


class KernelNet(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        bias: bool,
        omega_0: float,
    ):
        """
        Creates an 3-layer MLP that parameterizes a convolutional kernel as:

        relative position (1) -> hidden_channels (32) -> hidden_channels (32) -> out_channels (1) * in_channels (1)

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

        Activation = Sine
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
