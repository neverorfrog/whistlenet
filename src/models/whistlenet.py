import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from core.model import Classifier

DOMAIN_PARAMS = {"num_classes": 2, "input_channels": 1}

DATA_PARAMS = {
    "val_split_size": 0.15,
    "class_weights": [1, 30],
    "resample": True,
}

TRAIN_PARAMS = {
    "max_epochs": 20,
    "learning_rate": 0.000001,
    "batch_size": 64,
    "patience": 5,
    "metrics": "f1-score",
    "optim_function": torch.optim.Adam,
    "weight_decay": 0.0001,
    "loss_function": nn.NLLLoss(),
}

MODEL_PARAMS = {
    "channels": [1, 32, 64, 128, 64],
    "kernels": [5, 5, 5, 5],
    "strides": [2, 2, 2, 1],
    "pool_kernels": [2, 2, 2, 2],
    "pool_strides": [1, 1, 1, 1],
    "fc_dims": [3520, 2],
    "dropout": 0,
}


class WhistleNet(Classifier):
    def __init__(self, name, num_classes, bias=True):
        super().__init__(name, num_classes, bias=True)

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
            conv_layers.append(nn.LeakyReLU())
            conv_layers.append(
                nn.MaxPool1d(
                    kernel_size=pool_kernels[i], stride=pool_strides[i]
                )
            )
            conv_layers.append(nn.Dropout(MODEL_PARAMS["dropout"]))
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
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1], bias=bias))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(MODEL_PARAMS["dropout"]))
        self.fc = nn.Sequential(*fc_layers)

    @property
    def loss_function(self):
        return nn.CrossEntropyLoss()

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
