import torch
import torch.nn as nn

from core import MaxPool1dSame
from utils.focal import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from core.model import Model

MODEL_PARAMS = {
    "channels": [1, 32, 64, 128, 64],
    "kernels": [5, 5, 5, 5],
    "strides": [2, 2, 2, 1],
    "pool_kernels": [2, 2, 2, 2],
    "pool_strides": [1, 1, 1, 1],
    "fc_dims": [3520, 1],
}


class WhistleNet(Model):
    def __init__(self, config) -> None:
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
            conv_layers.append(nn.Dropout(config.model.dropout))

        conv_layers.append(
            nn.Conv1d(
                channels[-2],
                channels[-1],
                kernel_size=kernels[-1],
                stride=strides[-1],
                device=device,
            )
        )
        conv_layers.append(nn.Sigmoid())
        self.conv = nn.Sequential(*conv_layers)

        # Fully Connected layers
        fc_dims = MODEL_PARAMS["fc_dims"]
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1], bias=True))
            fc_layers.append(nn.Tanh())
        fc_layers.append(nn.Dropout(config.model.dropout))
        fc_layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_layers)

    @property
    def loss_function(self):
        return FocalLoss(gamma=2, alpha=0.5)

    @property
    def example_input(self):
        return (torch.randn(1, 1, 513),)

    def forward(self, x):
        """
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        """
        # Convolutional layers
        x = torch.tensor(x, dtype=torch.float32)
        x = self.conv(x)
        # print("state shape={0}".format(x.shape))

        # Fully Connected Layers
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def compute_score(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> float:
        binary_predictions = torch.where(
            predictions >= 0.5, torch.tensor(1), torch.tensor(0)
        )
        true_positives = (binary_predictions * labels).sum()
        false_positives = (binary_predictions * (1 - labels)).sum()
        false_negatives = ((1 - binary_predictions) * labels).sum()
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        if f1.isnan():
            f1 = 0.0
        return f1
