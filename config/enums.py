from enum import Enum

import torch.nn as nn
import torch.optim as optim
import torchmetrics

Optimizer = Enum(
    "Optimizer",
    {
        name: name
        for name in dir(optim)
        if not name.startswith("_") and callable(getattr(optim, name))
    },
)

Loss = Enum(
    "Loss",
    {
        name: name
        for name in dir(nn)
        if not name.startswith("_")
        and callable(getattr(nn, name))
        and "Loss" in name
    },
)
Loss = Enum("Loss", {**Loss.__members__, "FocalLoss": "FocalLoss"})

Metrics = Enum(
    "Metrics",
    {name: name for name in dir(torchmetrics) if not name.startswith("_")},
)

ActivationList = ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "Sine", "Swish"]
Activation = Enum(
    "ActivationFunction", {name: name for name in ActivationList}
)
