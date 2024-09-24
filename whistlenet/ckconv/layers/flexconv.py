import math

import torch

# typing
from omegaconf import OmegaConf

from config.config import WhistlenetConfig

from .ckconv import CKConvBase


class FlexConvBase(CKConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: WhistlenetConfig,
        separable: bool,
        **kwargs,
    ):
        self.data_dim = 1
        # Unpack mask_config values:
        mask_cfg = config.mask
        mask_type = mask_cfg.type
        mask_init_value = mask_cfg.init_value
        mask_learn_mean = mask_cfg.learn_mean
        mask_dynamic_cropping = mask_cfg.dynamic_cropping
        mask_threshold = mask_cfg.threshold

        if mask_type == "gaussian":
            init_spatial_value = mask_init_value * 1.667
        elif mask_type == "hann":
            init_spatial_value = mask_init_value
        else:
            raise NotImplementedError(
                f"Mask of type '{mask_type}' not implemented."
            )

        # Overwrite init_spatial value
        kernel_cfg = config.kernel
        kernel_cfg.init_spatial_value = init_spatial_value

        # call super class
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, config=config
        )

        # Define mask constructor
        self.mask_constructor = globals()[f"{mask_type}_mask"]
        # Define root finder & cropper functions
        root_function = f"{mask_type}_min_root"
        crop_function = self.crop_kernel_positions_causal
        self.root_function = globals()[root_function]
        self.crop_function = crop_function

        # Define learnable parameters of the mask
        mask_width_param = {
            "gaussian": {
                1: torch.Tensor([mask_init_value]),
                2: torch.Tensor([mask_init_value, mask_init_value]),
            },
        }[mask_type][self.data_dim]
        self.mask_width_param = torch.nn.Parameter(mask_width_param)

        mask_mean_param = {
            "gaussian": {
                1: torch.Tensor([1.0]),
                2: torch.Tensor([0.0, 0.0]),
            },
        }[mask_type][self.data_dim]
        if mask_learn_mean:
            self.mask_mean_param = torch.nn.Parameter(mask_mean_param)
        else:
            self.register_buffer("mask_mean_param", mask_mean_param)

        # Define threshold of mask for dynamic cropping
        mask_threshold = mask_threshold * torch.ones(1)
        self.register_buffer("mask_threshold", mask_threshold, persistent=True)

        # Save values in self
        self.dynamic_cropping = mask_dynamic_cropping

    def crop_kernel_positions_causal(
        self,
        kernel_pos: torch.Tensor,
        root: float,
    ):
        # In 1D, only one part of the array must be cut.
        if abs(root) >= 1.0:
            return kernel_pos
        else:
            # We not find the index from which the positions must be cropped
            # index = value - initial_linspace_value / step_size
            index = (
                torch.floor((root + 1.0) / self.linspace_stepsize).int().item()
            )  # TODO: zero?
            return kernel_pos[..., index:]

    def construct_masked_kernel(self, x):
        # Construct kernel
        # 1. Get kernel positions
        kernel_pos = self.handle_kernel_positions(x)
        # 2. dynamic cropping
        # Based on the current mean and sigma values, compute the [min, max] values of the array.
        with torch.no_grad():
            roots = self.root_function(
                thresh=self.mask_threshold,
                mean=self.mask_mean_param,
                sigma=self.mask_width_param,
                temperature=0,  # Only used for sigmoid
            )
            kernel_pos = self.crop_function(kernel_pos, roots)
        # 3. chang-initialize self.Kernel if not done yet.
        self.chang_initialization(kernel_pos)
        # 4. sample the kernel
        x_shape = x.shape
        conv_kernel = self.Kernel(kernel_pos).view(
            -1, x_shape[1], *kernel_pos.shape[2:]
        )
        # 5. construct mask and multiply with conv-kernel
        mask = self.mask_constructor(
            kernel_pos,
            self.mask_mean_param.view(1, -1, *(1,) * self.data_dim),
            self.mask_width_param.view(1, -1, *(1,) * self.data_dim),
            temperature=0.0,
        )
        self.conv_kernel = mask * conv_kernel
        # Return the masked kernel
        return self.conv_kernel


class FlexConv(FlexConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: WhistlenetConfig,
    ):
        # call super class
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            config=config,
            separable=False,
        )

    def forward(self, x):
        # 1. Compute the masked kernel
        conv_kernel = self.construct_masked_kernel(x)
        out = torch.nn.functional.conv1d(x, conv_kernel, self.bias, padding=0)
        return out


# ###############################
# # Gaussian Masks / Operations #
# ###############################
def gaussian_mask(
    kernel_pos: torch.Tensor,
    mask_mean_param: torch.Tensor,
    mask_width_param: torch.Tensor,
    **kwargs,
):
    # mask.shape = [1, 1, Y, X] in 2D or [1, 1, X] in 1D
    return torch.exp(
        -0.5
        * (
            1.0
            / (mask_width_param**2 + 1e-8)
            * (kernel_pos - mask_mean_param) ** 2
        ).sum(1, keepdim=True)
    )


def gaussian_inv_thresh(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    # Based on the threshold value, compute the value of the roots
    aux = sigma * torch.sqrt(-2.0 * torch.log(thresh))
    return torch.stack([mean - aux, mean + aux], dim=1)


def gaussian_min_root(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    return torch.min(gaussian_inv_thresh(thresh, mean, sigma))


def gaussian_max_abs_root(
    thresh: float,
    mean: float,
    sigma: float,
    **kwargs,
):
    return torch.max(
        torch.abs(gaussian_inv_thresh(thresh, mean, sigma)), dim=1
    ).values
