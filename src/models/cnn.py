import torch
import torch.nn as nn
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import transforms

from core.model import Model

DOMAIN_PARAMS = {"num_classes": 10, "input_channels": 1}

DATA_PARAMS = {
    "data_transform": transforms.Compose([transforms.ToTensor()]),
    "val_split_size": 0.15,
}

TRAIN_PARAMS = {
    "max_epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
    "patience": 5,
    "metrics": "f1-score",
    "optim_function": torch.optim.Adam,
    "weight_decay": 0.001,
    "loss_function": nn.NLLLoss(),
}

MODEL_PARAMS = {
    "channels": [DOMAIN_PARAMS["input_channels"], 10, 20],
    "kernels": [5, 5],
    "strides": [1, 1],
    "pool_kernels": [2, 2],
    "pool_strides": [2, 2],
    "fc_dims": [320, DOMAIN_PARAMS["num_classes"]],
    "dropout": 0.1,
}


class CNN(Model):
    def __init__(self, name) -> None:
        super().__init__(name)

        self.activation = nn.ReLU()

        # Convolutional Layers (take as input the image)
        channels = MODEL_PARAMS["channels"]
        kernels = MODEL_PARAMS["kernels"]
        strides = MODEL_PARAMS["strides"]
        pool_kernels = MODEL_PARAMS["pool_kernels"]
        pool_strides = MODEL_PARAMS["pool_strides"]

        conv_layers = []
        for i in range(len(kernels)):
            conv_layers.append(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    device=device,
                )
            )
            # conv_layers.append(nn.BatchNorm2d(channels[i+1]))
            conv_layers.append(self.activation)
            conv_layers.append(
                nn.MaxPool2d(
                    kernel_size=pool_kernels[i], stride=pool_strides[i]
                )
            )
        self.conv = nn.Sequential(*conv_layers)

        # Fully Connected layers
        fc_dims = MODEL_PARAMS["fc_dims"]
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.append(nn.Linear(fc_dims[i], fc_dims[i + 1], bias=True))
            fc_layers.append(self.activation)
        # fc_layers.append(nn.Dropout(MODEL_PARAMS['dropout']))
        self.fc = nn.Sequential(*fc_layers)

        # Initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         # Xavier/Glorot initialization for weights
        #         init.xavier_uniform_(m.weight)
        #         # Zero initialization for biases
        #         if m.bias is not None:
        #             init.zeros_(m.bias)

    @property
    def loss_function(self):
        return nn.CrossEntropyLoss()

    @property
    def example_input(self):
        return (torch.randn(1, 1, 28, 28),)

    def forward(self, x: torch.tensor):
        """
        Input:
            An image (already preprocessed)
        Output:
            Class of that image
        """
        # Convolutions
        x = x.type(torch.float).to(self.device)
        x = self.conv(x)
        # print("state shape={0}".format(x.shape))

        # Fully Connected Layers
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
