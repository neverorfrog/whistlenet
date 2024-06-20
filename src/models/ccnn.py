import torch
from omegaconf import OmegaConf

from ckconv import CKBlock
from core.model import Model


class CCNN(Model):
    def __init__(
        self, name: str, in_channels: int, out_channels: int, config: OmegaConf
    ):
        super().__init__(name)

        ckblocks = []
        for j in range(config.network.num_blocks):
            in_channels = (
                in_channels if j == 0 else config.network.hidden_channels
            )
            ckblock = CKBlock(
                in_channels, config.network.hidden_channels, config
            )
            ckblocks.append(ckblock)
        self.backbone = torch.nn.Sequential(*ckblocks)

        self.linear = torch.nn.Linear(
            config.network.hidden_channels, out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.linear(out[:, :, -1])

    @property
    def example_input(self):
        return (torch.randn(1, 1, 513),)  # hardcodato

    @property
    def loss_function(self):
        return torch.nn.CrossEntropyLoss()
