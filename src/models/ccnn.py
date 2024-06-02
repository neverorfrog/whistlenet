import torch
from omegaconf import OmegaConf

from ckconv import CKConv, LayerNorm, Linear1d
from ckconv.ckblock import CKBlock


class CCNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: OmegaConf):
        super().__init__()

        ckblocks = []
        for j in range(config.num_blocks):
            in_channels = in_channels if j == 0 else config.hidden_channels
            out_channels = config.hidden_channels
            ckblock = CKBlock(in_channels, out_channels, config)
            ckblocks.append(ckblock)
        self.backbone = torch.nn.Sequential(*ckblocks)

        self.linear = torch.nn.Linear(config.hidden_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.linear(out[:, :, -1])
