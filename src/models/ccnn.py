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

        self.linear = torch.nn.Linear(
            config.model.hidden_channels, out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.linear(out[:, :, -1])

    @property
    def example_input(self):
        return (torch.randn(1, 1, 513),)  # hardcodato

    @property
    def loss_function(self):
        return nn.CrossEntropyLoss()
