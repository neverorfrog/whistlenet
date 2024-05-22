import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.model import Classifier


class ToyModel(Classifier):
    def __init__(self, name, num_classes, bias=True) -> None:
        super().__init__(name, num_classes, bias)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    @property
    def loss_function(self):
        return nn.CrossEntropyLoss()

    @property
    def example_input(self) -> tuple:
        # return (torch.randn(1,256),)
        return (torch.randn(1, 1, 28, 28),)

    def forward(self, x) -> Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
