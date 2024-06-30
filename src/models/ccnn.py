import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ckconv import CKBlock
from core.model import Model
from utils import FocalLoss


class CCNN(Model):
    def __init__(self, in_channels: int, out_channels: int, config: OmegaConf):
        super().__init__(config)

        ckblocks = []
        for j in range(config.model.num_blocks):
            in_channels = (
                in_channels if j == 0 else config.model.hidden_channels
            )
            ckblock = CKBlock(
                in_channels, config.model.hidden_channels, config
            )
            ckblocks.append(ckblock)
        self.backbone = torch.nn.Sequential(*ckblocks)

        fc_layers = []
        fc_layers.append(
            torch.nn.Linear(config.model.hidden_channels, out_channels)
        )
        fc_layers.append(nn.Dropout(config.model.dropout))
        fc_layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.fc(out[:, :, -1])

    @property
    def example_input(self):
        return (torch.randn(1, 1, 513),)  # hardcodato

    @property
    def loss_function(self):
        return nn.BCELoss()

    def compute_score(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> float:
        binary_predictions = torch.where(
            predictions >= 0.5, torch.tensor(1), torch.tensor(0)
        )
        true_positives = (binary_predictions * labels).sum()
        false_positives = (binary_predictions * (1 - labels)).sum()
        if true_positives.sum() == 0 and false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        return precision
