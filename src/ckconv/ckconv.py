from typing import Optional, Tuple

import torch
import torch.nn.functional as f
from omegaconf import OmegaConf

from ckconv.kernelnet import KernelNet


def causal_padding(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Pad the input signal & kernel tensors.
    # Check if sizes are odd. If not, add a pad of zero to make them odd.
    if kernel.shape[-1] % 2 == 0:
        kernel = f.pad(kernel, (1, 0), value=0.0)
    # 2. Perform padding on the input so that output equals input in length
    x = f.pad(x, (kernel.shape[-1] - 1, 0), value=0.0)
    return x, kernel


class CKConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: OmegaConf):
        super().__init__()

        self.Kernel = KernelNet(
            out_channels * in_channels,
            config.kernel.hidden_channels,
            config.kernel.activation,
            config.kernel.bias,
            config.kernel.omega_0,
        )

        if config.kernel.bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        self.rel_positions = None
        self.register_buffer(
            "conv_kernel", torch.zeros(in_channels), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        rel_pos = self.handle_rel_positions(x)

        # Pass relative positions through self.Kernel (Step 2)
        conv_kernel = self.Kernel(rel_pos).view(-1, x_shape[1], *x_shape[2:])
        self.conv_kernel = conv_kernel

        # Compute the convolution (Step 3)
        x, kernel = causal_padding(x, conv_kernel)
        return torch.nn.functional.conv1d(x, kernel, self.bias, padding=0)

    def handle_rel_positions(self, x: torch.Tensor) -> torch.Tensor:
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
        if self.rel_positions is None:
            self.rel_positions = (
                torch.linspace(-1.0, 1.0, x.shape[-1])
                .unsqueeze(0)
                .unsqueeze(0)
            )
        return self.rel_positions
