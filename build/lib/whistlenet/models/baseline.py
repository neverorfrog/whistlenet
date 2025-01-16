import torch
import torch.nn as nn

from config.config import BaselineConfig
from whistlenet.core.layers import MaxPool1dSame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from whistlenet.core import Model
from whistlenet.core.utils import NUM_FREQS

MODEL_PARAMS = {
    "channels": [1, 32, 64, 128, 64],
    "kernels": [5, 5, 5, 5],
    "strides": [2, 2, 2, 1],
    "pool_kernels": [2, 2, 2, 2],
    "pool_strides": [2, 2, 2, 2],
    "fc_dims": [192, 1],
}


class Baseline(Model):
    """
    4 convolution blocks

    First Three Blocks

    - 1D Convolution
    - Batch normalization
    - ReLU
    - Max Pooling
    - Dropout

    This combination should ensure that the neural network reacts less strongly to loud noise

    Fourth Block

    - 1D Convolution
    - ELU
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__(config)

        # Convolutional Layers (take as input the image)
        channels = MODEL_PARAMS["channels"]
        kernels = MODEL_PARAMS["kernels"]
        strides = MODEL_PARAMS["strides"]
        pool_kernels = MODEL_PARAMS["pool_kernels"]
        pool_strides = MODEL_PARAMS["pool_strides"]

        conv_layers = []
        for i in range(3):
            conv_layers.append(
                nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    device=device,
                )
            )
            conv_layers.append(nn.BatchNorm1d(channels[i + 1]))
            conv_layers.append(nn.LeakyReLU(0.1))
            conv_layers.append(
                MaxPool1dSame(
                    kernel_size=pool_kernels[i], stride=pool_strides[i]
                )
            )
            if i > 0:
                conv_layers.append(nn.Dropout(config.hidden_dropout))
            else:
                conv_layers.append(nn.Dropout(config.dropout))

        conv_layers.append(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernels[-1],
                stride=strides[-1],
                device=device,
            )
        )
        conv_layers.append(nn.ELU())
        self.conv = nn.Sequential(*conv_layers)

        # Fully Connected layers
        fc_dims = MODEL_PARAMS["fc_dims"]
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1], bias=True))
            fc_layers.append(nn.Tanh())
        fc_layers.append(nn.Dropout(config.dropout))
        fc_layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_layers)

    @property
    def loss_function(self):
        return torch.nn.BCEWithLogitsLoss()

    @property
    def example_input(self):
        return (torch.randn(1, 1, NUM_FREQS),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        """
        # Convolutional layers
        x = x.clone().detach().to(device).requires_grad_(True)
        x = self.conv(x)
        # print("state shape={0}".format(x.shape))

        # Fully Connected Layers
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
