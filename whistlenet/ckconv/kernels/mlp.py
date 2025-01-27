import torch

import whistlenet.core.layers as layers
from config import KernelConfig
from whistlenet.core.utils import getcallable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KernelNet(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, config: KernelConfig
    ):
        """
        Creates an 3-layer MLP that parameterizes a convolutional kernel as:

        relative position (1) -> hidden_channels (32) -> hidden_channels (32) -> out_channels (1) * in_channels (1)

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            hidden_channels (int): The number of hidden channels.
            activation (str): The activation function to use.
            bias (bool): If True, adds a bias term to the convolution.
            omega_0 (float): The initial value of the kernel..

        Returns:
            torch.nn.Module: The created 3-layer MLP.
        """
        super().__init__()

        try:
            Activation: torch.nn.Module = getcallable(
                torch.nn, config.activation
            )
        except AttributeError:
            Activation: torch.nn.Module = getcallable(
                layers, config.activation
            )
        Linear: torch.nn.Module = getcallable(layers, config.linear_type)

        # The input of the network is a vector of relative positions and so input_dim = 1 for audio

        self.layers = []
        self.layers.extend(
            torch.nn.Sequential(
                Linear(1, config.hidden_channels, bias=config.bias),
                Activation(),
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
            )
        )
        self.hidden_layers = torch.nn.Sequential(*self.layers)

        self.output_linear = Linear(
            config.hidden_channels, out_channels, bias=config.bias
        )
        self.layers.extend(torch.nn.Sequential(self.output_linear))

        self.kernel_net = torch.nn.Sequential(*self.layers).to(device)
        self.initialize(Activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_net(x)

    def initialize(
        self,
        NonlinearType: torch.nn.Module,
    ):
        # Define the gain
        if NonlinearType == torch.nn.ReLU:
            nonlin = "relu"
        elif NonlinearType == torch.nn.LeakyReLU:
            nonlin = "leaky_relu"
        else:
            nonlin = "linear"

        # Initialize hidden layers
        for i, m in enumerate(self.hidden_layers.modules()):
            if isinstance(
                m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)
            ):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        torch.nn.init.kaiming_uniform_(
            self.output_linear.weight, nonlinearity="linear"
        )
        if self.output_linear.bias is not None:
            torch.nn.init.constant_(self.output_linear.bias, 0.0)
