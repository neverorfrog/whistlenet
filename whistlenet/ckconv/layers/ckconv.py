import math

import torch

import whistlenet.ckconv.kernels as kernels
from config.config import WhistlenetConfig
from whistlenet.core.utils import getcallable


class CKConvBase(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, config: WhistlenetConfig
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config

        self.Kernel: CKConvBase = getcallable(kernels, config.kernel.type)(
            in_channels, out_channels * in_channels, config.kernel
        )

        if config.kernel.bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        self.register_buffer(
            "train_length", torch.zeros(1).int(), persistent=False
        )
        self.register_buffer(
            "kernel_positions", torch.zeros(1), persistent=False
        )
        self.register_buffer(
            "conv_kernel", torch.zeros(in_channels), persistent=False
        )
        self.register_buffer(
            "linspace_stepsize", torch.zeros(1), persistent=False
        )
        self.register_buffer(
            "initialized", torch.zeros(1).bool(), persistent=False
        )

    def construct_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass consists of 3 main steps:

        1. The very first time the CKConv sees an input, it will check its
           dimensionality. Based on that, we create a vector of relative
           positions of equal length.

        2. We now pass the vector of relative positions through self.Kernel.
           This will create the convolutional kernel of the layer on the fly.

        3. We then compute the convolution.
        """

        # Construct kernel (Step 1)
        x_shape = x.shape
        kernel_pos = self.handle_kernel_positions(x)
        self.chang_initialization(kernel_pos)

        # Pass relative positions through self.Kernel (Step 2)
        conv_kernel = self.Kernel(kernel_pos).view(
            -1, x_shape[1], *kernel_pos.shape[2:]
        )
        self.conv_kernel = conv_kernel
        return self.conv_kernel

    def handle_kernel_positions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Responsible for generating the relative positions of the input, given
        its dimensionality.

        Parameters
        ----------
        x : torch.tensor
            The input to the layer.

        Returns
        -------
        rel_pos : torch.tensor
            The relative positions of the input of form (batch_size=1,
            in_channels=1, x_dimension=x.shape[-1]). This means there are Nx
            relative positions (where Nx is the length of the input) between -1
            and 1.
        """

        if self.kernel_positions.shape[-1] == 1:
            if self.train_length[0] == 0:
                if self.config.kernel.size == -1:
                    self.train_length[0] = x.shape[-1]
                elif int(self.config.kernel.size) % 2 == 1:
                    # Odd number
                    self.train_length[0] = int(self.config.kernel.size)
                else:
                    raise ValueError(
                        f"The size of the kernel must be either 'full', 'same' or an odd number"
                        f" in string format. Current: {self.config.kernel.size}"
                    )

            self.kernel_positions = (
                torch.linspace(-1.0, 1.0, self.train_length[0])
                .unsqueeze(0)
                .unsqueeze(0)
            )

        return self.kernel_positions

    def chang_initialization(self, kernel_positions):
        if not self.initialized[0]:
            # Initialization - Initialize the last layer of self.Kernel as in Chang et al. (2020)
            with torch.no_grad():
                kernel_size = kernel_positions.reshape(
                    *kernel_positions.shape[:-1], -1
                )
                kernel_size = kernel_size.shape[-1]
                normalization_factor = self.in_channels * kernel_size
                self.Kernel.output_linear.weight.data *= math.sqrt(
                    1.0 / normalization_factor
                )
            # Set the initialization flag to true
            self.initialized[0] = True


class CKConv(CKConvBase):
    def __init__(
        self, in_channels: int, out_channels: int, config: WhistlenetConfig
    ):
        super().__init__(in_channels, out_channels, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_kernel = self.construct_kernel(x)
        output = torch.nn.functional.conv1d(
            x, conv_kernel, self.bias, padding=0
        )
        return output
