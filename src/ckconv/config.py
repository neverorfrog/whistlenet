import torch

from ckconv.expression import Sine, Swish
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
